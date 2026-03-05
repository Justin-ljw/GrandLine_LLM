"""
SFT 训练脚本：由 pretrain.py 复制后做少量修改得到，便于与预训练对比讲解。
改动处均用 # [SFT] 标出。

主要差异：
1. 数据集：PretrainDataset(.bin) → SFTDataset(jsonl 对话数据，只算 assistant loss)
2. Tokenizer：SFT 需提前加载（Dataset 需要），pretrain 仅评测时加载
3. 模型加载：pretrain 用 from_pretrained，SFT 用 load_state_dict 加载 .pth
4. 评测方式：pretrain 用 C3/XCOPA benchmark，SFT 用 mini_bench + DeepSeek Judge 生成式评测
5. 新增参数：tokenizer_path, judge_api_key, judge_model
"""
import os
import sys

# 禁用 tokenizers 并行警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist  # [DDP] 多进程/多 GPU 通信；without_ddp 无此 import
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel  # [DDP] DDP 包模型；without_ddp 无
from torch.utils.data import DataLoader, DistributedSampler  # [DDP] 多卡用 DistributedSampler；without_ddp 仅 DataLoader
from transformers import AutoTokenizer  # [SFT] SFT 需一开始就加载 tokenizer（给 SFTDataset 用），pretrain 仅在 eval_bench=1 时加载
from dataset.sft_dataset import SFTDataset  # [SFT] pretrain 用 PretrainDataset(.bin)，SFT 用 SFTDataset(jsonl + 只算 assistant loss)
from model.config import GrandLineConfig
from model.model_grandline import GrandLineForCausalLM

from utils import get_lr, Logger, is_main_process, init_distributed_mode, SkipBatchSampler  # [DDP] is_main_process/init_distributed_mode 仅 DDP 用；without_ddp 无
from benchmark.mini_bench.eval import run_inference, run_judge_async  # [SFT] mini_bench 评测函数，推理 + Judge；pretrain 用 run_benchmark (C3/XCOPA)

warnings.filterwarnings('ignore')


# 与 pretrain 基本相同
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
        
        # 保存 checkpoint [DDP] 仅主进程写盘；without_ddp 无 is_main_process() 判断
        if (global_step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            
            # [DDP] DDP 下取 .module 才是真实模型；without_ddp 仅 getattr(model, '_orig_mod', model)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
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

        # [SFT] 自定义 mini_bench 评测(LLM as Judge)：每到 eval_interval 用当前模型推理，异步调用 DeepSeek 进行 Judge
        # pretrain 用 run_benchmark (C3/XCOPA)，SFT 用 run_inference + run_judge_async (生成式评测)
        # Benchmark 评测 [DDP] 仅主进程跑；without_ddp 无 is_main_process() 判断
        if args.eval_bench == 1 and tokenizer is not None and global_step % args.eval_interval == 0 and is_main_process():
            # 解包获取原始模型
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            raw = getattr(raw, "_orig_mod", raw)
            #开启评测模式
            model.eval()
            # 给模型输入 mini_bench 的 prompt ，让模型生成回答
            pairs = run_inference(raw, tokenizer, device=args.device, num_samples=3)
            # 模型推理完成就可以继续训练了，Judge 的评测在后台异步进行，不会阻塞训练
            model.train()
            
            valid_dir = os.path.join(full_save_dir, "valid_samples")
            valid_file = os.path.join(valid_dir, f"global_step_{global_step}.jsonl")
            # 异步调用 LLM Judge 评测，传入模型生成的回答和自己的 LLM Judge API Key
            run_judge_async(pairs, args.judge_api_key, args.judge_model,
                           output_file=valid_file,
                           swanlab_log_fn=swanlab.log if swanlab else None,
                           global_step=global_step)
            Logger(f'[eval] step={global_step} 推理完成，Judge 后台运行中...')

        # 清理本 epoch 的变量，释放显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GrandLine SFT Training")
    # [SFT] 以下参数默认值与 pretrain.py 不同：save_dir, save_weight, epochs, data_path, from_weight, swanlab_project
    # [SFT] 新增参数：tokenizer_path, ollama_url, ollama_model
    parser.add_argument("--save_dir", type=str, default="../model_weight/sft/exp_1", help="模型保存根目录")
    parser.add_argument('--save_weight', default='sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数（SFT 推荐 2-3 epoch，过多会过拟合）")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="初始学习率（SFT 推荐 1e-5 ~ 1e-4，从预训练继续可用 5e-5）")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="warmup 占比（SFT 推荐更长 warmup，0.1 即 10%）")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减系数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="", help="SFT 数据 jsonl 路径")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer_15k", help="tokenizer 路径")  # [SFT] 新增：SFTDataset 需要 tokenizer
    parser.add_argument('--from_weight', default='', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="GrandLine-SFT", help="swanlab项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--eval_bench", default=1, type=int, choices=[0, 1], help="是否评测benchmark（0=否，1=是）")
    parser.add_argument("--eval_interval", type=int, default=1000, help="每隔多少 step 跑 mini_bench（0=关闭），用当前模型推理+DeepSeek Judge 打分")
    # [SFT] 新增：mini_bench 评测参数（使用 DeepSeek API）
    parser.add_argument("--judge_api_key", type=str, default='', help="Judge API Key（可直接传入或从环境变量 DEEPSEEK_API_KEY 读取）")
    parser.add_argument("--judge_model", type=str, default="deepseek-chat", help="Judge 模型名")
    args = parser.parse_args()


    # ========== 1. [DDP] 初始化分布式环境 ==========
    # without_ddp 无此步骤，直接进入配置目录
    local_rank = init_distributed_mode()  # 多卡时初始化进程组并返回本卡 GPU 号
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"  # DDP 时每进程用不同 GPU

    # ========== 2. 配置目录、检查 checkpoint ==========
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
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配置swanlab ==========
    swanlab_run = None
    if args.use_swanlab == 1 and is_main_process():  # [DDP] 仅主进程上报；without_ddp 无 is_main_process()
        import swanlab
        # 传自己的 API Key
        swanlab.login(api_key="")
        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            id=swanlab_id,
            resume=True,
            config=vars(args)
        )
        Logger(f'SwanLab initialized: {run_name}')
    
    # ========== 5. 定义模型、数据、优化器 ==========
    lm_config = GrandLineConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    
    # [SFT] 先加载 tokenizer（SFT 数据集需要，pretrain 仅在 eval_bench=1 时加载）
    Logger(f'Loading tokenizer from {args.tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 创建或加载模型
    # [SFT] 用 load_state_dict 加载 .pth 权重文件，pretrain 用 from_pretrained 加载模型目录
    if args.from_weight.lower() != 'none' and os.path.exists(args.from_weight):
        Logger(f'Loading model from {args.from_weight}')
        model = GrandLineForCausalLM(lm_config)
        model.load_state_dict(torch.load(args.from_weight, map_location='cpu'), strict=False)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = GrandLineForCausalLM(lm_config)
        
    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 使用 torch.compile 加速模型前向和反向传播
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # [SFT] 数据集：SFTDataset 读取 jsonl 对话数据，只计算 assistant 部分的 loss
    # pretrain 用 PretrainDataset 读取预处理好的 .bin 文件，计算全部 token 的 loss
    Logger('Loading dataset...')
    train_ds = SFTDataset(jsonl_path=args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    # [DDP] 多卡用 DistributedSampler 分片；without_ddp 无 train_sampler，后面用 indices
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    Logger('Dataset ready')
    
    # 梯度缩放器和优化器
    Logger('Initializing optimizer...')
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    Logger('Optimizer ready')
    
    # ========== 6. 从ckp恢复状态 ==========
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
    
    # ========== 7. [DDP] DDP 包模型 ==========
    # without_ddp 无此整段，不包 DDP
    if dist.is_initialized():
        Logger('Wrapping model with DDP...')
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger('DDP ready')

    # ========== 8. 计算总步数 ==========
    # [DDP] 多卡时除以 world_size；without_ddp 为 len(train_ds) // args.batch_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_ds) // (args.batch_size * world_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_percent)  # 10% warmup（SFT 推荐更长 warmup）
    Logger(f'World size: {world_size}, Steps per epoch: {steps_per_epoch}')
    Logger(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps} (10%)')
    
    # ========== 8.5. 初始评测 (step 0) ==========
    # [DDP] 仅主进程评测；without_ddp 无 is_main_process()
    if args.eval_bench == 1 and tokenizer is not None and is_main_process() and start_epoch == 0 and start_step == 0:
        Logger('Running initial mini_bench evaluation (step 0)...')
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, "_orig_mod", raw)
        model.eval()
        pairs = run_inference(raw, tokenizer, device=args.device, num_samples=3)
        model.train()
        
        valid_dir = os.path.join(full_save_dir, "valid_samples")
        valid_file = os.path.join(valid_dir, f"global_step_0.jsonl")
        run_judge_async(pairs, args.judge_api_key, args.judge_model,
                       output_file=valid_file,
                       swanlab_log_fn=swanlab_run.log if swanlab_run else None,
                       global_step=0)
        Logger('[eval] step=0 初始评测完成，Judge 后台运行中...')
    
    # ========== 9. 开始训练 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # [DDP] 多卡时打乱各卡分片；without_ddp 无此行
        # 用 epoch 固定种子，保证续训时同一 epoch 的打乱顺序与初次训练完全一致
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(len(train_ds), generator=g).tolist()
        
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # [DDP] 多卡用 train_sampler，单卡用 indices；without_ddp 仅 SkipBatchSampler(indices, ...)
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        Logger(f'Creating DataLoader for epoch {epoch+1}...')
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f'DataLoader ready, starting epoch {epoch+1}...')
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
    
    # ========== 10. [DDP] 清理分布式进程组 ==========
    # without_ddp 无此步骤，仅 Logger('Training done.')
    if dist.is_initialized(): dist.destroy_process_group()
    
    Logger('Training done.')
