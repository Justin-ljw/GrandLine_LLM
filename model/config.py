from transformers import PretrainedConfig

class GrandLineConfig(PretrainedConfig):
    model_type = "grandline"
    
    """
    GrandLine 模型配置类
    
    参数说明：
        hidden_size: 隐藏层维度
        num_hidden_layers: Transformer-Decoder 层数
        num_attention_heads: 注意力Query头数
        num_key_value_heads: KV 头数（用于 Grouped Query Attention）
        intermediate_size: FFN 中间层维度
        vocab_size: 词表大小
        num_experts: MoE专家数量
        max_position_embeddings: 最大序列长度
        rope_theta: RoPE 基础频率
        hidden_act: 激活函数类型
        dropout: Dropout 比例
        weight_decay: 权重衰减系数
        rms_norm_eps: RMSNorm 的 epsilon
        bos_token_id: 开始 token ID
        eos_token_id: 结束 token ID
        flash_attn: 是否启用 Flash Attention
    """
    
    def __init__(
        self,
        # 模型核心架构参数（0.1B 标准配置）
        hidden_size: int = 768, 
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 2048,
        vocab_size: int = 15000,
        
        # 位置编码
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        
        # 激活函数和正则化
        hidden_act: str = 'silu',
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-05,
        
        # 特殊token
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        
        # flash注意力机制
        flash_attn: bool = True,
        **kwargs):
            super().__init__(**kwargs)
            
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.intermediate_size = intermediate_size
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            self.hidden_act = hidden_act
            self.dropout = dropout
            self.rms_norm_eps = rms_norm_eps
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.flash_attn = flash_attn

    
    
    