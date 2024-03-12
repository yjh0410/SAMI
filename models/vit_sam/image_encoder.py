# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Vision Transformer of Segment-Anything ----------------------
class ImageEncoderViT(nn.Module):
    """
    We remove the neck which used in the Segment-Anything.
    """
    def __init__(self,
                 img_size            : int = 1024,
                 patch_size          : int = 16,
                 in_chans            : int = 3,
                 embed_dim           : int = 768,
                 depth               : int = 12,
                 num_heads           : int = 12,
                 mlp_ratio           : float = 4.0,
                 qkv_bias            : bool = True,
                 norm_layer          : Type[nn.Module] = nn.LayerNorm,
                 act_layer           : Type[nn.Module] = nn.GELU,
                 use_abs_pos         : bool = True,
                 use_rel_pos         : bool = False,
                 window_size         : int = 0,
                 global_attn_indexes : Tuple[int, ...] = (),
                 checkpoint = None
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed: Optional[nn.Parameter] = None
        self.checkpoint = checkpoint
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        # ------------ Model parameters ------------
        ## Patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ## ViT blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim         = embed_dim,
                          num_heads   = num_heads,
                          mlp_ratio   = mlp_ratio,
                          qkv_bias    = qkv_bias,
                          norm_layer  = norm_layer,
                          act_layer   = act_layer,
                          use_rel_pos = use_rel_pos,
                          window_size = window_size if i not in global_attn_indexes else 0,
                          input_size  = (img_size // patch_size, img_size // patch_size),
                          )
            self.blocks.append(block)

        self.load_pretrained()

    def load_pretrained(self):
        if self.checkpoint is not None:
            print('Loading SAM pretrained weight from : {}'.format(self.checkpoint))
            # checkpoint state dict
            checkpoint_state_dict = torch.load(self.checkpoint, map_location="cpu")
            # model state dict
            model_state_dict = self.state_dict()
            encoder_state_dict = {}
            # check
            for k in list(checkpoint_state_dict.keys()):
                if "image_encoder" in k and k[14:] in model_state_dict:
                    shape_model = tuple(model_state_dict[k[14:]].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model == shape_checkpoint or "pos_embed" in k:
                        encoder_state_dict[k[14:]] = checkpoint_state_dict[k]
                    else:
                        print("Shape unmatch: ", k)
 
            # interpolate position embedding
            interpolate_pos_embed(self, encoder_state_dict)

           # load the weight
            self.load_state_dict(encoder_state_dict, strict=False)
        else:
            print('No SAM pretrained.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # [B, H, W, C] -> [B, N, C]
        return x.flatten(1, 2)


# ---------------------- Model modules ----------------------
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU,
                 ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        
        return x

class Block(nn.Module):
    def __init__(self,
                 dim               : int,
                 num_heads         : int,
                 mlp_ratio         : float = 4.0,
                 qkv_bias          : bool = True,
                 norm_layer        : Type[nn.Module] = nn.LayerNorm,
                 act_layer         : Type[nn.Module] = nn.GELU,
                 use_rel_pos       : bool = False,
                 window_size       : int = 0,
                 input_size        : Optional[Tuple[int, int]] = None,
                 ) -> None:
        super().__init__()
        # -------------- Basic parameters --------------
        self.window_size = window_size
        # -------------- Model parameters --------------
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim               = dim,
                              num_heads         = num_heads,
                              qkv_bias          = qkv_bias,
                              use_rel_pos       = use_rel_pos,
                              input_size        = input_size if window_size == 0 else (window_size, window_size),
                              )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 use_rel_pos: bool = False,
                 input_size: Optional[Tuple[int, int]] = None,
                 ) -> None:
        super().__init__()
        # -------------- Basic parameters --------------
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

        # -------------- Model parameters --------------
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self,
                 kernel_size : Tuple[int, int] = (16, 16),
                 stride      : Tuple[int, int] = (16, 16),
                 padding     : Tuple[int, int] = (0, 0),
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        return x


# ---------------------- Model functions ----------------------
def window_partition(x: torch.Tensor,
                     window_size: int,
                     ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor,
                       window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int],
                       ) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
        
    return x

def get_rel_pos(q_size: int,
                k_size: int,
                rel_pos: torch.Tensor,
                )-> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn : torch.Tensor,
                           q    : torch.Tensor,
                           rel_pos_h : torch.Tensor,
                           rel_pos_w : torch.Tensor,
                           q_size    : Tuple[int, int],
                           k_size    : Tuple[int, int],
                           ) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        # Pos embed from checkpoint
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # Pos embed from model
        pos_embed_model = model.pos_embed
        num_patches = model.num_patches
        # [B, H, W, C] -> [B, N, C]
        pos_embed_checkpoint = pos_embed_checkpoint.flatten(1, 2)
        pos_embed_model = pos_embed_model.flatten(1, 2)
        
        orig_num_postions = pos_embed_model.shape[-2]
        num_extra_tokens  = orig_num_postions - num_patches

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size  = int(num_patches ** 0.5)

        # height (== width) for the new position embedding
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("- Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens,
                                                         size=(new_size, new_size),
                                                         mode='bicubic',
                                                         align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed = new_pos_embed.reshape(-1, int(orig_num_postions ** 0.5), int(orig_num_postions ** 0.5), embedding_size)
            checkpoint_model['pos_embed'] = new_pos_embed


# ------------------------ Model Functions ------------------------
def build_vit_sam(model_name="vit_h", img_size=224, patch_size=16, img_dim=3, checkpoint=None):
    if model_name == "vit_b":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=768,
                               depth=12,
                               num_heads=12,
                               mlp_ratio=4.0,
                               act_layer=nn.GELU,
                               checkpoint=checkpoint,
                               )
    if model_name == "vit_l":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=1024,
                               depth=24,
                               num_heads=16,
                               mlp_ratio=4.0,
                               act_layer=nn.GELU,
                               checkpoint=checkpoint,
                               )
    if model_name == "vit_h":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               embed_dim=1280,
                               depth=32,
                               num_heads=16,
                               mlp_ratio=4.0,
                               act_layer=nn.GELU,
                               checkpoint=checkpoint,
                               )
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, c, h, w = 2, 3, 224, 224
    x = torch.randn(bs, c, h, w)
    patch_size = 16

    # Build model
    model = build_vit_sam(patch_size=patch_size)
    if torch.cuda.is_available():
        x = x.cuda()
        model.cuda()

    # Inference
    outputs = model(x)
    print(outputs.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
