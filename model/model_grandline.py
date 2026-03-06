import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union
from .config import GrandLineConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int , eps: float = 1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# RoPE 位置编码
def precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6):
    """
        预计算 RoPE (Rotary Position Embedding) 的 cos 和 sin 频率
        
        Args:
            dim: 注意力头的维度 (head_dim)
            end: 最大序列长度
            rope_base: RoPE 的基础频率，默认 1e6
        
        Returns:
            freqs_cos: cos 频率张量 (end, dim)
            freqs_sin: sin 频率张量 (end, dim)
    """
    
    # 计算频率：θ_i = base^(-2i/d), i ∈ [0, d/2)
    freqs = 1.0 / rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    
    # 生产位置序列
    pos = torch.arange(end, device=freqs.device)
    # 计算外积(一列 * 一行)得到频率矩阵(shape: [end, dim // 2])：freqs[pos, i] = pos * θ_i
    freqs = torch.outer(pos, freqs).float()
    
    # 计算cos和sin值，并复制一次（同一组的2个值要用同一个cos和sin）
    # (这里的分组实际是(i, i + dim // 2)，而不是(i, i + 1))
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
        应用 RoPE (Rotary Position Embedding) 到 query 和 key
        
        Args:
            q: Query 张量 (batch, seq_len, num_heads, head_dim)
            k: Key 张量 (batch, seq_len, num_kv_heads, head_dim)
            cos: 预计算的 cos 频率
            sin: 预计算的 sin 频率
            unsqueeze_dim: 用于广播的维度
        
        Returns:
            q_embed: 应用 RoPE 后的 query
            k_embed: 应用 RoPE 后的 key
    """
    
    def rotate_half(x):
        # 将张量分成两半，并交换位置:[x1, x2] -> [-x2, x1]
        return torch.cat([-x[..., x.shape[-1] // 2: ], x[..., : x.shape[-1] // 2]], dim=-1)
    
    # cos 和 sin 的 shape 是 (seq_len, head_dim), 传参时先把freqs_cos和freqs_sin切片到x的seq_len再传入
    # cos.unsqueeze(unsqueeze_dim) 的 shape 变为 (seq_len, 1, head_dim)，可以广播到 q 和 k 的 shape
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed


def repeat_key_value(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对 KV 进行重复以匹配 Query 的头数（用于 Grouped Query Attention）
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效，torch.repeat_interleave会真的复制数据。
    而下面的实现只是使用torch.expand让多个指针指向同一份存储空间来实现重复的效果，避免了不必要的数据复制
    
    Args:
        x: KV 张量 (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 重复次数 (num_heads // num_kv_heads)
    
    Returns:
        重复后的张量 (batch, seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 先在 num_key_value_heads 维度上增加一个维度，再在这个新维度上重复 n_rep 次，最后 reshape 到 (batch, seq_len, num_attention_heads, head_dim)
    return x[:, :, :, None, :].expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)


# 注意力机制实现，支持 Grouped Query Attention（GQA）
class Attention(nn.Module):
    def __init__(self, args: GrandLineConfig):
        super().__init__()
        
        self.hidden_size = args.hidden_size
        # query头数（也是输出头数）
        self.num_attention_heads = args.num_attention_heads
        # key和value的头数，如果没有单独指定 num_key_value_heads，则默认为 num_attention_heads（即不分组，做MHA而不是GQA）
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        
        # 每个注意力头的维度
        assert self.hidden_size % self.num_attention_heads == 0, 'GQA: hidden_size 必须能被 num_attention_heads 整除！'
        self.head_dim = self.hidden_size // self.num_attention_heads
        # 进行分组查询注意力（GQA）时，kv重复的次数
        assert self.num_attention_heads % self.num_key_value_heads == 0, 'GQA: num_attention_heads 必须能被 num_key_value_heads 整除！'
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        
        # QKV投影层
        self.Wq = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Dropout
        self.dropout = args.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Flash Attention 支持检测
        self.flash_attn = hasattr(nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        
    def forward(
        self, 
        x: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False, 
        attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, hidden_size)
            position_embeddings: (cos, sin) RoPE 位置编码
            past_key_value: KV cache，用于推理加速
            use_cache: 是否返回新的 KV cache，训练时为 False，推理时为 True
            attention_mask: 注意力掩码 (batch_size, num_attention_heads, seq_len, seq_len)，1=有效位置，0=padding
        
        Returns:
            output: 注意力输出 (batch, seq_len, hidden_size)
            past_kv: 新的 KV cache（如果 use_cache=True）
        """
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)
        
        # 分头处理：将最后一个维度切分成 (num_attention_heads, head_dim) 或 (num_key_value_heads, head_dim)
        # xq 的 shape 变为 (batch_size, seq_len, num_attention_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
        # 如果提供了 past_key_value，则将其与当前的 kry 和 value 拼接起来（在 seq_len 维度）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果 use_cache=True，则组织并返回新的 past_key_value 供下一步使用；否则返回 None
        past_key_value = (xk, xv) if use_cache else None
        
        xq, xk, xv = (
            xq.transpose(1, 2), 
            repeat_key_value(xk, self.n_rep).transpose(1, 2), 
            repeat_key_value(xv, self.n_rep).transpose(1, 2)
        )
        
        # 使用 Flash Attention (PyTorch >= 2.0) - 仅在训练时且无 KV cache 时启用
        # （ Flash Attention 未显示支持 KV Cache ，若要用 KV Cache 要手动实现 Attention ）
        # Flash Attention 是 pytorch 官方在2.0版本引入的一种高效的注意力实现，本质是实现了底层算子优化，Attention的计算原理不变
        if self.flash_attn and (seq_len > 1) and (past_key_value is None):
            
            # 预训练时满 token ，无 padding，直接使用 is_causal=True（最快路径）
            if attention_mask is None or torch.all(attention_mask == 1):
                attn_output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=True
                    )
                
            # 有 padding， Flash Attention 要构造 Boolean mask (True=可见, False=屏蔽)
            else:
                # Causal mask: 下三角为 True（可见），上三角为 False（屏蔽）
                causal_mask = torch.tril(
                    torch.ones((seq_len, seq_len), device=xq.device, dtype=torch.bool),
                    diagonal=0 
                )
                
                # Padding mask:  (batch, seq_len) -> (batch, 1, 1, seq_len)
                # 1 -> True (可见), 0 -> False (padding，屏蔽)
                padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=bool)  # (batch, 1, 1, seq_len)
                
                # 组合 mask: 两个都为 True 才能参与运算（逻辑与）
                # causal_mask: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                # padding_mask: (batch, 1, 1, seq_len) -> broadcast 到每个 query 位置
                combined_mask = causal_mask.unsqueeze(0) & padding_mask  # (batch, 1, seq_len, seq_len)
                
                attn_output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=combined_mask, 
                    dropout_p=self.dropout if self.training else 0.0
                )
                
                
        # 传统 Attention 实现（用于推理时的 KV cache 或 PyTorch < 2.0）
        else:
            scores = (torch.matmul(xq, xk.transpose(-1, -2)) / math.sqrt(self.head_dim))
            
            # 应用 causal mask（上三角设为 -inf）
            # scorse 的 shape 是 (batch, num_attention_heads, q_len, k_len)，
            # 其中 k_len 可能大于 q_len（因为有 past_key_value 的拼接），但我们只需要在 (q_len x k_len) 的右上角(q_len, q_len)部分应用 mask
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=scores.device), diagonal=1)
            
            # padding mask（如果提供了 attention_mask，则在 scores 上添加 -inf 来屏蔽掉 padding 位置）
            if attention_mask is not None:
                # atention_mask 的shape本来为(batch_size, k_len)，且用0/1标识是否为 padding token，0为 padding token
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, k_len)，插入两维度方便后续广播
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 将 1/0 转换为 0/-inf
                scores = scores + extended_attention_mask
                
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, xv)  # (batch, num_attention_heads, q_len, head_dim)
        
        # 将多头合并回 (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.Wo(attn_output))
        
        return output, past_key_value


class FeedForward(nn.Module):
    """
    前馈神经网络（SwiGLU 激活函数）
    结构: Gate(x) * Up(x) -> Down
    """
    def __init__(self, args: GrandLineConfig):
        super().__init__()
        
        self.hidden_size = args.hidden_size
        
        self.intermediate_size = args.intermediate_size
        # 计算中间层大小：若为指定则默认为 hidden_size * 8/3，向上取整到 64 的倍数
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 8 / 3)
            self.intermediate_size = 64* ((self.intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # ACT2FN 可实现根据 config 动态切换激活函数而不用改代码
        self.act_fn = ACT2FN[args.hidden_act]
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        """SwiGLU: act(gate_proj(x)) * up_proj(x) -> down_proj"""
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class GrandLineBlock(nn.Module):
    """
    Transformer-Decoder 块：Masked-Self-Attention + FeedForward
    采用 Pre-Norm 结构（Norm before attention/mlp）
    """
    def __init__(self, layer_id: int, args: GrandLineConfig):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size
        self.rms_norm_eps = args.rms_norm_eps
        
        self.self_attn = Attention(args)
        self.feed_forward = FeedForward(args)
        self.input_rms_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.post_attn_rms_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        
    def forward(
        self, 
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播：Pre-Norm Transformer Block
        
        结构：
            x = x + Attention(Norm(x))
            x = x + MLP(Norm(x))
        """
        # Attention 部分
        residual = x
        x, presnet_key_value = self.self_attn(self.input_rms_norm(x), 
                                              position_embeddings, 
                                              past_key_value, 
                                              use_cache, 
                                              attention_mask)
        x = x + residual
        
        # FeedForward 部分
        x = x + self.feed_forward(self.post_attn_rms_norm(x))
         
        return x, presnet_key_value


class GrandLineModel(nn.Module):
    """
    GrandLine 模型主体（Decoder-only Transformer）
    """
    def __init__(self, config: GrandLineConfig):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Token Embedding
        self.embed_token = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Decoder Blocks
        self.layers = nn.ModuleList([GrandLineBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终的 LayerNorm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算 RoPE 频率（注册为 buffer，不参与更新但会跟着模型）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads, 
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)
        
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None, 
                **kwargs):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)，1=有效位置，0=padding
            past_key_values: KV cache 列表，用于推理加速
            use_cache: 是否返回新的 KV cache
        
        Returns:
            hidden_states: 最后一层的隐藏状态 (batch, seq_len, hidden_size)
            present_key_values: 新的 KV cache 列表
        """
        _, seq_len = input_ids.shape
        
        # 兜底处理：如果 past_key_values 是一个 Cache 对象，而不是一个 List ，则将其置为 None ，即不使用 KV_Cache ，避免后续代码出错
        if hasattr(past_key_values, 'layers'):
            past_key_values =  None
        # 如果 past_key_values 是 None，则创建一个长度为 num_hidden_layers 的列表，给每一层准备一个 None 占位
        past_key_values = past_key_values or [None] * self.num_hidden_layers
        
        # 新输入的token的位置编码起始位置（如果有 past_key_values （推理）则从上次的 k_len 位置开始，否则（训练）从0开始）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_len], 
            self.freqs_sin[start_pos: start_pos + seq_len]
            )
        
        # 把输入的 token_ids 转换为嵌入向量，并应用 dropout
        hidden_states = self.dropout(self.embed_token(input_ids))
        
        # 逐层处理 Transformer Block，并收集新的 KV cache
        present_key_values = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_value = layer(hidden_states,
                                                    position_embeddings,
                                                    past_key_value=past_key_value,
                                                    use_cache=use_cache,
                                                    attention_mask=attention_mask)
            # 将每一层的新的 KV cache 添加到列表中，供下一步返回
            present_key_values.append(present_key_value)
            
        hidden_states = self.final_norm(hidden_states)
        return hidden_states, present_key_values
    

class GrandLineForCausalLM(PreTrainedModel, GenerationMixin):
    """
    包含语言模型头的 GrandLine 因果语言模型，用于文本生产任务
    在 GrandLineModel 基础上添加 Language Modeling Head
    """
    config_class = GrandLineConfig
    
    def __init__(self, config: GrandLineConfig = None):
        self.config = config or GrandLineConfig()
        super().__init__(self.config)
        
        # Transformer 主体
        self.model = GrandLineModel(self.config)
        
        # language modeling head，与 token embedding 共享权重
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_token.weight = self.lm_head.weight  # 权重共享
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """
        为 generation 处理输入：如果有过去计算的 KV cache，则只输入最后一个 token
        """
        # 如果 past_key_values 存在，说明之前已经把前面的 token 缓存了
        if past_key_values is not None:
            # 只有最后生成的那个 token 需要送入模型，大大节省算力
            input_ids = input_ids[:, -1:]
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }    
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None, 
                logits_to_keep: Union[int, torch.Tensor] = 0, 
                **args):
        """
        前向传播（用于训练和推理）
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)
            labels: 标签 (batch, seq_len)，用于计算 loss
            past_key_values: KV cache
            use_cache: 是否返回 KV cache
            logits_to_keep: 保留最后多少个 token 的 logits（节省内存）
        
        Returns:
            CausalLMOutputWithPast: 包含 loss, logits, past_key_values, hidden_states
        """
        # Transformer 主体前向传播，得到最后的隐藏状态和新的 KV cache
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, 
            past_key_values=past_key_values, 
            use_cache=use_cache, 
            attention_mask=attention_mask,
            **args
        )
        
        # 计算 logits（可选择只保留最后几个 token， 默认 logits_to_keep = 0 即全部 token 的输出都保留，即训练模式）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 计算 loss（如果提供了 labels ，即训练，则计算交叉熵损失；否则，为推理 loss 为 None）
        loss = None
        if labels is not None:
            # 标准的自回归语言模型 loss 计算：
            # 使用 token[0:i] 的信息，预测 token[i+1]
            # shift_logits: [0, 1, ..., n-2] 位置的预测（去掉eos）
            # shift_labels: [1, 2, ..., n-1] 位置的真实标签（去掉bos ，并左移）
            shift_logits = logits[..., : -1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()  # token id 序列
            
            # 把 logits 和 labels 展平到二维和一维，计算交叉熵损失，ignore_index=-100 用于忽略 padding 和 prompt 的位置
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100  # 忽略 padding 和 prompt 的位置
            )
        
        # 把数据组织成类输出，包含 loss, logits, past_key_values, hidden_states
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        
        return output
