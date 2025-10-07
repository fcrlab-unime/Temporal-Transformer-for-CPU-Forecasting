import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    # [L, L] with -inf above diagonal
    mask = torch.full((L, L), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class SpatialEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] tokens are VMs
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        h = self.norm2(x)
        h = self.ff2(F.gelu(self.ff1(h)))
        x = x + self.drop2(h)
        return x


class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B*N, T, D] tokens are time steps per VM; causal mask over T
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        h = self.norm2(x)
        h = self.ff2(F.gelu(self.ff1(h)))
        x = x + self.drop2(h)
        return x


class ForecastingTransformer(nn.Module):
    def __init__(
        self,
        num_vms: int,
        num_features: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 192,
        num_heads_spatial: int = 6,
        num_heads_temporal: int = 6,
        ff_dim: int = 256,
        num_layers_spatial: int = 3,
        num_layers_temporal: int = 3,
        dropout: float = 0.1,
        use_vm_embed: bool = True,
        use_time_embed: bool = True,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vms = num_vms
        self.num_features = num_features
        self.d_model = d_model
        self.grad_ckpt = grad_ckpt

        self.vm_feature_proj = nn.Linear(num_features, d_model)
        self.vm_pos_embedding = nn.Parameter(torch.randn(1, num_vms, d_model)) if use_vm_embed else None

        self.time_pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model)) if use_time_embed else None

        self.spatial_layers = nn.ModuleList([
            SpatialEncoderLayer(d_model, num_heads_spatial, ff_dim, dropout) for _ in range(num_layers_spatial)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalEncoderLayer(d_model, num_heads_temporal, ff_dim, dropout) for _ in range(num_layers_temporal)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * num_features),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.vm_feature_proj.weight)
        nn.init.zeros_(self.vm_feature_proj.bias)
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, F]
        returns: y [B, pred_len, N, F]
        """
        B, T, N, Fdim = x.shape
        assert N == self.num_vms and Fdim == self.num_features, "Input dims mismatch"

        device = x.device


        h = self.vm_feature_proj(x)


        if self.vm_pos_embedding is not None:
            h = h + self.vm_pos_embedding[:, :N, :].unsqueeze(1) 


        spatial_out = []
        for t in range(T):
            ht = h[:, t, :, :] 
            for layer in self.spatial_layers:
                if self.grad_ckpt and self.training:
                    ht = torch.utils.checkpoint.checkpoint(layer, ht, use_reentrant=False)
                else:
                    ht = layer(ht)
            spatial_out.append(ht)
        h_spatial = torch.stack(spatial_out, dim=1) 

        if self.time_pos_embedding is not None:
            h_spatial = h_spatial + self.time_pos_embedding[:, :T, :].unsqueeze(2)

        causal_mask = generate_causal_mask(T, device) 
        h_temporal = h_spatial.permute(0, 2, 1, 3).contiguous() 
        h_temporal = h_temporal.view(B * N, T, self.d_model) 

        for layer in self.temporal_layers:
            if self.grad_ckpt and self.training:
                h_temporal = torch.utils.checkpoint.checkpoint(layer, h_temporal, causal_mask, use_reentrant=False)
            else:
                h_temporal = layer(h_temporal, causal_mask)

        last_token = h_temporal[:, -1, :] 

        decoded = self.decoder(last_token) 
        y = decoded.view(B, N, self.pred_len, self.num_features)  
        y = y.permute(0, 2, 1, 3).contiguous() 
        return y


if __name__ == "__main__":
    # sanity check
    B, T, N, F = 4, 24, 32, 3
    model = ForecastingTransformer(
        num_vms=N,
        num_features=F,
        seq_len=T,
        pred_len=6,
        d_model=192,
        num_heads_spatial=6,
        num_heads_temporal=6,
        ff_dim=256,
        num_layers_spatial=2,
        num_layers_temporal=2,
        dropout=0.1,
        use_vm_embed=True,
        use_time_embed=True,
        grad_ckpt=False,
    )
    x = torch.randn(B, T, N, F)
    y = model(x)
    print("out:", y.shape)  # [B, pred_len, N, F]
