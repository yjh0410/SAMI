import math
import torch
import torch.nn as nn

try:
    from vit_modules import ViTBlock
except:
    from .vit_modules import ViTBlock


# ------------------------ Vision Transformer ------------------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size    :int    = 224,
                 patch_size  :int    = 16,
                 img_dim     :int    = 3,
                 emb_dim     :int    = 768,
                 num_layers  :int    = 12,
                 num_heads   :int    = 12,
                 qkv_bias    :bool   = True,
                 mlp_ratio   :float  = 4.0,
                 dropout     :float  = 0.,
                 num_classes :int    = 1000,
                 learnable_pos :bool = True):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.img_dim = img_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        self.learnable_pos = learnable_pos
        # -------- network parameters --------
        ## vit encoder
        self.patch_embed = nn.Conv2d(img_dim, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.ModuleList([ViTBlock(emb_dim, qkv_bias, num_heads, mlp_ratio, dropout) for _ in range(num_layers)])
        self.norm        = nn.LayerNorm(emb_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim)) if learnable_pos else None
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, emb_dim))
        ## classifier
        self.classifier  = nn.Linear(emb_dim, num_classes)

    def _init_weight(self,):
        # initialize cls_token
        nn.init.normal_(self.cls_token, std=1e-6)
        # initialize pos_embed
        if self.learnable_pos:
            nn.init.normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_posembed(self, x, cls_token=False, temperature=10000):
        scale = 2 * math.pi
        embed_dim, grid_h, grid_w = x.shape[1:]
        num_pos_feats = embed_dim // 2
        # get grid
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(grid_h, dtype=torch.float32, device=x.device),
             torch.arange(grid_w, dtype=torch.float32, device=x.device)])
        # normalize grid coords
        y_embed = y_embed / (grid_h + 1e-6) * scale
        x_embed = x_embed / (grid_w + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        # [H, W, C] -> [N, C]
        pos_embed = torch.cat((pos_y, pos_x), dim=-1).view(-1, embed_dim)
        if cls_token:
            # [1+N, C]
            pos_embed = torch.cat([torch.zeros([1, embed_dim], device=pos_embed.device), pos_embed], dim=0)

        return pos_embed.unsqueeze(0)

    def forward(self, x):
        # patch embed
        x = self.patch_embed(x)

        # get position embedding
        if self.learnable_pos:
            pos_embed = self.pos_embed
        else:
            pos_embed = self.get_posembed(x, cls_token=True)

        # reshape: [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # transformer
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        x += pos_embed
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        # classify
        return self.classifier(x[:, 0, :])


# ------------------------ Model Functions ------------------------
def vit_nano(img_size=224, patch_size=16, img_dim=3, num_classes=1000, dropout=0., learnable_pos=False):
    model = VisionTransformer(img_size      = img_size,
                              patch_size    = patch_size,
                              img_dim       = img_dim,
                              emb_dim       = 192,
                              num_layers    = 12,
                              num_heads     = 8,
                              mlp_ratio     = 4.0,
                              dropout       = dropout,
                              num_classes   = num_classes,
                              learnable_pos = learnable_pos)

    return model

def vit_tiny(img_size=224, patch_size=16, img_dim=3, num_classes=1000, dropout=0., learnable_pos=False):
    model = VisionTransformer(img_size      = img_size,
                              patch_size    = patch_size,
                              img_dim       = img_dim,
                              emb_dim       = 384,
                              num_layers    = 12,
                              num_heads     = 8,
                              mlp_ratio     = 4.0,
                              dropout       = dropout,
                              num_classes   = num_classes,
                              learnable_pos = learnable_pos)

    return model

def vit_base(img_size=224, patch_size=16, img_dim=3, num_classes=1000, dropout=0., learnable_pos=False):
    model = VisionTransformer(img_size      = img_size,
                              patch_size    = patch_size,
                              img_dim       = img_dim,
                              emb_dim       = 768,
                              num_layers    = 12,
                              num_heads     = 12,
                              mlp_ratio     = 4.0,
                              dropout       = dropout,
                              num_classes   = num_classes,
                              learnable_pos = learnable_pos)

    return model

def vit_large(img_size=224, patch_size=16, img_dim=3, num_classes=1000, dropout=0., learnable_pos=False):
    model = VisionTransformer(img_size      = img_size,
                              patch_size    = patch_size,
                              img_dim       = img_dim,
                              emb_dim       = 1024,
                              num_layers    = 24,
                              num_heads     = 16,
                              mlp_ratio     = 4.0,
                              dropout       = dropout,
                              num_classes   = num_classes,
                              learnable_pos = learnable_pos)

    return model

def vit_huge(img_size=224, patch_size=16, img_dim=3, num_classes=1000, dropout=0., learnable_pos=False):
    model = VisionTransformer(img_size      = img_size,
                              patch_size    = patch_size,
                              img_dim       = img_dim,
                              emb_dim       = 1280,
                              num_layers    = 32,
                              num_heads     = 16,
                              mlp_ratio     = 4.0,
                              dropout       = dropout,
                              num_classes   = num_classes,
                              learnable_pos = learnable_pos)

    return model


if __name__ == '__main__':
    import torch
    from ptflops import get_model_complexity_info

    # build model
    model = vit_tiny(patch_size=16)

    # calculate params & flops
    flops_count, params_count = get_model_complexity_info(model,(3,224,224), as_strings=True, print_per_layer_stat=False)

    print('flops: ', flops_count)
    print('params: ', params_count)
    