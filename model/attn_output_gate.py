import torch.nn as nn
import torch
from config import GrandLineConfig

class AttnOutputGate(nn.Module):
    def __init__(self, config: GrandLineConfig):
        super().__init__()
        self.config = config
        
        self.gate_type = getattr(config, 'attn_gate_type', 'none')
        self.init_bias = getattr(config, 'attn_gate_init_bias', 4.0)
        
        # 根据不同的 Gate 粒度调整 W_g 的维度
        if self.gate_type == 'token':
            out_dim = 1
        elif self.gate_type == 'head':
            out_dim = self.config.num_attention_heads
        elif self.gate_type == 'channel':
            out_dim = self.config.num_attention_heads * self.config.head_size
        
        self.W_g = nn.Linear(self.config.hidden_size, out_dim, bias=True)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, self.init_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
            x: (batch_size, seq_len, hidden_size)
            attn_output: (batch_size, num_attn_head, seq_len, head_size)
            
            'token': (batch_size, seq_len, 1) -> (batch_size, 1, seq_len, 1)
            'head': (batch_size, seq_len, num_attn_head) -> (batch_size, num_attn_head, seq_len, 1)
            'chanel': (batch_size, seq_len, num_attn_head * head_size) -> (batch_size, num_attn_head, seq_len, head_size)
        """
        batch_size, seq_len, _ = x.shape
        
        gate = torch.sigmoid(self.W_g(x))
        
        # 根据不同的 Gate 粒度，让gate与attn_output形状适配
        if self.gate_type == 'token':
            gate = gate.squeeze(-1)[:, None, :, None]
        elif self.gate_type == 'head':
            gate = gate.permute(0, 2, 1).unsqueeze(-1)
        elif self.gate_type == 'channel':
            gate = gate.view(batch_size, seq_len, self.config.num_attention_heads, self.config.head_size).permute(0, 2, 1, 3)
        
        return gate