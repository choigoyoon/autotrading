import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for HierarchicalTradingTransformer"""
    feature_dims: Dict[str, int]
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    max_seq_len: int = 100
    
    def to_dict(self):
        return self.__dict__
        
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = nn.GELU()
        
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        residual = x
        x = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.dropout(x)
        x = self.self_attn_layer_norm(residual + x)
        
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)
        
        return x

class TimeFrameEncoder(nn.Module):
    """Encodes a single timeframe's sequence data"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, config.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.layer_norm(x)
        return x

class HierarchicalTradingTransformer(nn.Module):
    """A hierarchical Transformer model for multi-timeframe trading data"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Timeframe-specific encoders
        self.timeframe_encoders = nn.ModuleDict({
            tf: TimeFrameEncoder(config, config.feature_dims[tf])
            for tf in config.timeframes
        })
        
        # Hierarchical attention
        self.hierarchical_attention = MultiHeadAttention(config)
        
        # Output heads
        self.direction_head = nn.Linear(config.d_model, 3) # Buy, Sell, Hold
        self.magnitude_head = nn.Linear(config.d_model, 1)
        self.duration_head = nn.Linear(config.d_model, 1)
        self.confidence_head = nn.Linear(config.d_model, 1)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        timeframe_outputs = []
        for tf in self.config.timeframes:
            if tf in x:
                encoded = self.timeframe_encoders[tf](x[tf])
                # Use the final hidden state of the sequence
                timeframe_outputs.append(encoded[:, -1, :])
        
        # Stack outputs for hierarchical attention
        hierarchical_input = torch.stack(timeframe_outputs, dim=1)
        
        # Apply hierarchical attention
        fused_representation = self.hierarchical_attention(
            query=hierarchical_input,
            key=hierarchical_input,
            value=hierarchical_input
        )
        
        # Use the mean of the fused representation for final prediction
        final_representation = fused_representation.mean(dim=1)
        
        # Get predictions from output heads
        direction = self.direction_head(final_representation)
        magnitude = torch.sigmoid(self.magnitude_head(final_representation)) # 0-1 range
        duration = torch.relu(self.duration_head(final_representation)) # Must be positive
        confidence = torch.sigmoid(self.confidence_head(final_representation)) # 0-1 range
        
        return {
            'direction': direction,
            'magnitude': magnitude.squeeze(-1),
            'duration': duration.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }
