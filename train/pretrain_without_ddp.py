"""
单卡预训练脚本（无 DDP）
用法: python pretrain_without_ddp.py [args]
"""
import os
import sys

import torch.amp

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁止tokenizers库的并行化，避免死锁

__package__ = 'train'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 将项目根目录添加到sys.path



import argparse
import time
import warnings
from contextlib import nullcontext
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset.pretrain_dataset import PretrainDataset
from model.config import GrandLineConfig
from model.model_grandline import GrandLineForCausalLM
from utils import Logger, get_lr, SkipBatchSampler
from benchmark.evaluator import run_benchmark

_BENCH_PRETRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, total_steps=None, warmup_steps=None, full_save_dir=None):
    '''
    训练一个 epoch 的函数，包含前向传播、反向传播、日志打印、checkpoint 保存和 benchmark 评测
    
    Args:
        epoch: 当前 epoch 索引
        loader: 从 start_step 开始的 DataLoader 对象，提供训练数据
        iters: 当前 epoch 的总迭代次数（steps）
        start_step: 当前 epoch 已经完成的 step 数（用于断点续训）
        swanlab: SwanLab 运行对象（可选，用于日志记录）
        total_steps: 预训练的总步数（用于学习率调度）
        warmup_steps: 学习率预热的步数（用于学习率调度）
        full_save_dir: 模型 checkpoint 的保存目录（用于保存模型权重）
    '''
    start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)
        
        current_step = epoch * iters + step
        
        # 根据当前 step 调节学习率，并设置到优化器中
        lr = get_lr(
            lr=args.learning_rate, 
            current_step=current_step, 
            total_steps=total_steps, 
            warmup_steps=warmup_steps)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 自动混合精度训练 搭配 梯度缩放防止下溢
        with autocast_ctx:
            res = model(input_ids=input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps
            
        # 梯度放大防止下溢
        scaler.scale(loss).backward()
        
        # 梯度累积以实现更大的有效 batch size
        if (step + 1) % args.accumulation_steps == 0:
            # 把梯度从放大状态恢复到正常状态，并更新模型参数
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸，减缓训练波动
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            # 自适应调节缩放系数
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        global_step = epoch * iters + step
        
        # 定期打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), Loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            
            if swanlab:
                swanlab.log({"loss": current_loss, "learning_rate": current_lr, "eta_time": eta_min}, step=global_step)
        
        # 保存 checkpoint
        if global_step % args.save_interval == 0 or step == iters - 1:
            model.eval()
            
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            
            raw_model = getattr(model, '_orig_mod', model)
            # 把权重参数转换为 FP16 来保存
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, f'{ckp_dir}/{args.save_weight}_{lm_config.hidden_size}.pth')
            # 同时保存一个 resume.pth，方便后续断点续训时自动加载
            torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'global_step': global_step,
                'swanlab_id': getattr(swanlab, 'id', None) if swanlab else None
            }, f'{ckp_dir}/resume.pth')
            
            Logger(f'Checkpoint saved at step {global_step} to {ckp_dir}')
            model.train()

        # Benchmark 评测
        if args.eval_bench == 1 and tokenizer is not None and (global_step % args.eval_interval == 0):
            model.eval()
            # benchmark 的数据路径
            c3_path = os.path.join(_BENCH_PRETRAIN_DIR, "clue_c3_eval_500.jsonl")
            xcopa_path = os.path.join(_BENCH_PRETRAIN_DIR, "xcopa_zh_merged.jsonl")
            eval_results = run_benchmark(model=model, tokenizer=tokenizer, c3_path=c3_path, xcopa_path=xcopa_path)
            
            if swanlab:
                swanlab.log(eval_results, step=global_step)
            Logger(f'Benchmark evaluated at step {global_step}: {eval_results}')
            
            model.train()

        # 清理本 epoch 的变量，释放显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GrandLine Pretraining (Single GPU)")
    parser.add_argument("--save_dir", type=str, default="../model_weight/pretrain/exp_single_1", help="模型保存根目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--warmup_percent", type=float, default=0.03, help="warmup 占比 3%")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减系数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=3000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="", help="预处理后的.bin文件路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="GrandLine-Pretrain", help="swanlab项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--eval_bench", default=1, type=int, choices=[0, 1], help="是否评测benchmark（0=否，1=是）")
    parser.add_argument("--eval_interval", type=int, default=1000, help="评测间隔步数")
    args = parser.parse_args()
    
    
    # ========== 1. 配置目录、检查 checkpoint ==========
    # 生成 run_name（用于后续创建子目录）
    run_name = f"h{args.hidden_size}_l{args.num_hidden_layers}_bs{args.batch_size}_lr{args.learning_rate}"
    full_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 断点续训时自动检测最新 checkpoint 并加载
    ckp_data = None
    if args.from_resume == 1:
        ckp_dirs = [d for d in os.listdir(full_save_dir) if d.startswith('global_step_')]
        if ckp_dirs:
            latest_ckp = max(ckp_dirs, key=lambda x: int(x.split('_')[-1]))
            resume_path = f'{full_save_dir}/{latest_ckp}/resume.pth'
            if os.path.exists(resume_path):
                ckp_data = torch.load(resume_path, map_location='cpu')
                Logger(f'Found checkpoint: {full_save_dir}/{latest_ckp}')
    
    # ========== 2. 混合精度 ==========
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 3. SwanLab ==========
    swanlab_run = None
    if args.use_swanlab == 1:
        import swanlab
        # 传自己的 API Key
        swanlab.login(api_key='')
        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            id=swanlab_id,
            resume=True,
            config=vars(args)
        )
        Logger(f'SwanLab initialized: {run_name}')
        
    # ========== 4. 模型、数据、优化器 ==========
    lm_config = GrandLineConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    
    # 创建或加载模型
    if args.from_weight.lower() != 'none' and os.path.exists(args.from_weight):
        Logger(f'Loading model from {args.from_weight}')
        model = GrandLineForCausalLM.from_pretrained(args.from_weight)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = GrandLineForCausalLM(lm_config)
    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 创建 tokenizer 用于评测 benchmark
    if args.eval_bench == 1:
        from transformers import AutoTokenizer
        # 评测时需要 tokenizer 来处理输入文本，但预训练阶段不直接使用（预训练数据已在 dataset 中处理）
        # 传模型的 tokenizer 的路径
        tokenizer = AutoTokenizer.from_pretrained('tokenizer_15k')
        Logger('Tokenizer loaded for benchmark evaluation')
    else:
        tokenizer = None
    
    # 使用 torch.compile 加速模型前向和反向传播
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        
    # 创建数据集
    Logger('Loading dataset...')
    train_ds = PretrainDataset(data_path=args.data_path, seq_len=args.max_seq_len)
    Logger('Dataset ready')
    
    # 梯度缩放器和优化器
    Logger('Initializing optimizer...')
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    Logger('Optimizer ready')
    
    # ========== 5. 从 checkpoint 恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        Logger('Loading checkpoint...')
        if args.use_compile == 1:
            raw_model = getattr(model, '_orig_mod', model)
            raw_model.load_state_dict(ckp_data['model'])
        else:
            model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Checkpoint loaded: epoch={start_epoch}, step={start_step}')
        
    # ========== 6. 总步数（单卡）==========
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_percent)  # 3% 的 warmup
    Logger(f'Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}')
    
    # ========== 7. 初始评测 (step 0) ==========
    if args.eval_bench == 1 and tokenizer is not None and start_epoch == 0 and start_step == 0:
        Logger('Running initial benchmark evaluation (step 0)...')
        model.eval()
        # benchmark 的数据路径
        c3_path = os.path.join(_BENCH_PRETRAIN_DIR, "clue_c3_eval_500.jsonl")
        xcopa_path = os.path.join(_BENCH_PRETRAIN_DIR, "xcopa_zh_merged.jsonl")
        eval_results = run_benchmark(model=model, tokenizer=tokenizer, c3_path=c3_path, xcopa_path=xcopa_path)
        
        if swanlab_run:
            swanlab_run.log(eval_results, step=0)
        Logger(f'Initial benchmark results (step 0): {eval_results}')
        
        model.train()
        
    # ========== 8. 训练循环 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size} (single GPU)')
    for epoch in range(start_epoch, args.epochs):
        # 用 epoch 固定种子，保证续训时同一 epoch 的打乱顺序与初次训练完全一致
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(len(train_ds), generator=g).tolist()
        
        # 使用 SkipBatchSampler 处理 DataLoader ，跳过前 start_step 个 batch ，适用于断点续训
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(sampler=indices, batch_size=args.batch_size, skip_batches=skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(
                epoch=epoch, 
                loader=loader, 
                iters=len(loader) + skip, 
                start_step=start_step, 
                swanlab=swanlab_run, 
                total_steps=total_steps, 
                warmup_steps=warmup_steps, 
                full_save_dir=full_save_dir
                )
        else:
            train_epoch(
                epoch=epoch, 
                loader=loader, 
                iters=len(loader), 
                start_step=0, 
                swanlab=swanlab_run, 
                total_steps=total_steps, 
                warmup_steps=warmup_steps, 
                full_save_dir=full_save_dir
                )
    
    Logger('Training done.')
