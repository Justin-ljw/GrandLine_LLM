from .config import ModelConfig

class RMSNormParams:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def total_params(self):
        return self.config.hidden_size
    
class AttentionParams:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def total_params(self):
        Wq_params = self.config.hidden_size * self.config.num_attention_heads * self.config.head_size
        Wk_params = self.config.hidden_size * self.config.num_key_value_heads * self.config.head_size
        Wv_params = self.config.hidden_size * self.config.num_key_value_heads * self.config.head_size
        Wo_params = self.config.hidden_size * self.config.num_attention_heads * self.config.head_size
        
        if self.config.attn_gate_type == 'none':
            gate_params = 0
        elif self.config.attn_gate_type == 'token':
            gate_params = self.config.hidden_size
        elif self.config.attn_gate_type == 'head':
            gate_params = self.config.head_size * self.config.num_attention_heads
        elif self.config.attn_gate_type == 'channel':
            gate_params = self.config.head_size * self.config.num_attention_heads * self.config.head_size
        
        return Wq_params + Wk_params + Wv_params + Wo_params + gate_params
    
class FFNParams:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def total_params(self):
        up_proj_params = self.config.hidden_size * self.config.intermediate_size
        gate_proj_params = self.config.hidden_size * self.config.intermediate_size
        down_proj_params = self.config.hidden_size * self.config.intermediate_size
        
        return up_proj_params + gate_proj_params + down_proj_params
    
class TransformerBlockParams:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        self.pre_attn_norm_params = RMSNormParams(config)
        self.attn_params = AttentionParams(config)
        self.pre_ffn_norm_params = RMSNormParams(config)
        self.ffn_params = FFNParams(config)
        
    def total_params(self):
        return self.pre_ffn_norm_params.total_params() + self.attn_params.total_params() + self.pre_ffn_norm_params.total_params() + self.ffn_params.total_params()
    
class EmbeddingParams:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def total_params(self):
        return self.config.vocab_size * self.config.hidden_size
    
class TransformerModelProfile:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        self.embedding_params = EmbeddingParams(config)
        self.transformer_block_params = TransformerBlockParams(config)
        self.final_norm_params = RMSNormParams(config)
        
    def total_params(self):
        return self.embedding_params.total_params() + self.config.num_hidden_layers * self.transformer_block_params.total_params() + self.final_norm_params.total_params()