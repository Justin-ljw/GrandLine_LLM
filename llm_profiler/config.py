
class ModelConfig:
    model_type = "grandline"
    
    """
    模型配置类
    
    参数说明：
        hidden_size: 隐藏层维度
        num_hidden_layers: Transformer-Decoder 层数
        num_attention_heads: 注意力Query头数
        num_key_value_heads: KV 头数（用于 Grouped Query Attention）
        head_size: 每个头的维度
        intermediate_size: FFN 中间层维度
        vocab_size: 词表大小
        attn_gate_type: gate attn的粒度，只能是("none", "token", "head", "channel")之一，其中"none"表示不使用gate
    """
    
    def __init__(
        self,
        # 模型核心架构参数（0.1B 标准配置）
        hidden_size: int = 768, 
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_size: int = 64, 
        intermediate_size: int = 2048,
        vocab_size: int = 15000,
        attn_gate_type: str = 'none',
        
        **kwargs):
            super().__init__(**kwargs)
            
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError("num_attention_heads 必须被 num_key_value_heads 整除")
            
            self.head_size = head_size
            self.intermediate_size = intermediate_size
            self.vocab_size = vocab_size
            
            # Gated Attention
            if attn_gate_type not in ("none", "token", "head", "channel"):
                raise ValueError(f'Invalid attn_gate_type={attn_gate_type}')
            self.attn_gate_type = attn_gate_type

    
    
    