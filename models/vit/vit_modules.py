import torch
import torch.nn as nn


# ------------------------ Basic Modules ------------------------
## Multi-Layer Perceptron
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self,
                 in_dim     :int,
                 hidden_dim :int,
                 out_dim    :int,
                 drop       :float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

## Vanilla Multi-Head Attention
class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 dropout   :float = 0.):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)


    def forward(self, x):
        bs, N, _ = x.shape
        # ----------------- Input proj -----------------
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # ----------------- Multi-head Attn -----------------
        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        q = q.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        ## [B, H, Nq, C_h] X [B, H, C_h, Nk] = [B, H, Nq, Nk]
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # [B, H, Nq, C_h]

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

## ViT's Block
class ViTBlock(nn.Module):
    def __init__(
            self,
            dim       :int,
            qkv_bias  :bool  = False,
            num_heads :int   = 8,
            mlp_ratio :float = 4.0,
            dropout   :float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

