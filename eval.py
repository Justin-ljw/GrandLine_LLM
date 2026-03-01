"""
GrandLine 模型交互式对话脚本
"""
import argparse
import torch
from transformers import AutoTokenizer, TextStreamer
from model.config import GrandLineConfig
from model.model_grandline import GrandLineForCausalLM

def main():
    parser = argparse.ArgumentParser(description="GrandLine模型交互对话")
    parser.add_argument('--model_path', default='', type=str, help="模型权重路径（.pth文件）")
    parser.add_argument('--tokenizer_path', default='./tokenizer_15k', type=str, help="Tokenizer路径")
    parser.add_argument('--model_type', default='sft', type=str, choices=['pretrain', 'sft'], help="模型类型：pretrain（文本续写）或 sft（对话）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.2, type=float, help="生成温度（0-1）")
    parser.add_argument('--top_p', default=0.7, type=float, help="nucleus采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--multi_turn', action='store_true', help="保留对话历史（多轮）；不传则单轮，每轮独立")
    args = parser.parse_args()
    
    # 自动推断模型类型（从文件名）
    if 'pretrain' in args.model_path:
        args.model_type = 'pretrain'
    elif 'sft' in args.model_path:
        args.model_type = 'sft'
    
    # 加载模型和tokenizer
    print(f'加载模型: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    model = GrandLineForCausalLM(GrandLineConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval().to(args.device)
    
    print(f'✅ 模型加载完成！设备: {args.device}')
    print(f'📝 模型类型: {args.model_type} ({"对话模式" if args.model_type == "sft" else "文本续写"})')
    print(f'📎 对话模式: {"多轮（保留历史）" if args.multi_turn else "单轮（每轮独立）"}\n')
    print('='*60)
    print('💬 开始对话 (输入 exit 退出)')
    print('='*60)
    
    conversation = []  # 仅 multi_turn 时使用
    # 流式输出，减少用户等待感（每生成一个token就输出）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    
    while True:
        user_input = input('\n👤 你: ').strip()
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            print('👋 再见！')
            break
        
        if not user_input:
            continue
        
        if args.model_type == 'pretrain':
            formatted_input = user_input
            conversation = []
        else:
            # SFT：按是否多轮决定是否保留历史
            if args.multi_turn:
                conversation.append({"role": "user", "content": user_input})
            else:
                conversation = [{"role": "user", "content": user_input}]
            # SFT 模式要把输入输出转化为问答格式
            formatted_input = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # tokenizer 编码输入
        inputs = tokenizer(formatted_input, return_tensors="pt").to(args.device)
        
        # 生成回复
        print('💫 GrandLine: ', end='', flush=True)
        with torch.no_grad():
            # 自回归生成
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.2  # 重复惩罚系数：> 1 时会惩罚重复生成的token，鼓励模型生成更多样化的回复
            )
        
        # 解码生成的回复（去掉输入部分）
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=False
        )
        # 把当前回复加入 conversation（仅多轮 SFT 模式），以便下一轮输入时能看到历史对话
        if args.model_type == 'sft' and args.multi_turn:
            conversation.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
