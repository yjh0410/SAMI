import math
import torch
import torch.nn as nn

try:
    from vit_modules import ViTBlock
except:
    from .vit_modules import ViTBlock


# ------------------------ Basic Modules ------------------------
## Masked Image Modeling (MIM) ViT Encoder
class MAE_ViT_Encoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 224,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 768,
                 en_num_layers :int   = 12,
                 en_num_heads  :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 mask_ratio    :float = 0.75):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.img_dim = img_dim
        self.en_emb_dim = en_emb_dim
        self.en_num_layers = en_num_layers
        self.en_num_heads = en_num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        # -------- network parameters --------
        ## vit encoder
        self.patch_embed = nn.Conv2d(img_dim, en_emb_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.ModuleList([ViTBlock(en_emb_dim, qkv_bias, en_num_heads, mlp_ratio, dropout) for _ in range(en_num_layers)])
        self.norm        = nn.LayerNorm(en_emb_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, en_emb_dim), requires_grad=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, en_emb_dim))

        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_posembed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(pos_embed)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():           
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_posembed(self, embed_dim, grid_size, cls_token=False, temperature=10000):
        scale = 2 * math.pi
        grid_h, grid_w = grid_size, grid_size
        num_pos_feats = embed_dim // 2
        # get grid
        y_embed, x_embed = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32),
                                           torch.arange(grid_w, dtype=torch.float32)])
        # normalize grid coords
        y_embed = y_embed / (grid_h + 1e-6) * scale
        x_embed = x_embed / (grid_w + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
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
            pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)

        return pos_embed.unsqueeze(0)

    def random_masking(self, x):
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original position of each patch

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get th binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

## Masked ViT Decoder
class MAE_ViT_Decoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 784,
                 de_emb_dim    :int   = 512,
                 de_num_layers :int   = 12,
                 de_num_heads  :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 norm_pix_loss :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.en_emb_dim = en_emb_dim
        self.de_emb_dim = de_emb_dim
        self.de_num_layers = de_num_layers
        self.de_num_heads = de_num_heads
        self.norm_pix_loss = norm_pix_loss
        # -------- network parameters --------
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
        self.decoder_embed     = nn.Linear(en_emb_dim, de_emb_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, de_emb_dim), requires_grad=False)  # fixed sin-cos embedding
        self.transformer       = nn.ModuleList([ViTBlock(de_emb_dim, qkv_bias, de_num_heads, mlp_ratio, dropout) for _ in range(de_num_layers)])
        self.decoder_norm      = nn.LayerNorm(de_emb_dim)
        self.decoder_pred      = nn.Linear(de_emb_dim, patch_size**2 * img_dim, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = self.get_posembed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():           
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_posembed(self, embed_dim, grid_size, cls_token=False, temperature=10000):
        scale = 2 * math.pi
        grid_h, grid_w = grid_size, grid_size
        num_pos_feats = embed_dim // 2
        # get grid
        y_embed, x_embed = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32),
                                           torch.arange(grid_w, dtype=torch.float32)])
        # normalize grid coords
        y_embed = y_embed / (grid_h + 1e-6) * scale
        x_embed = x_embed / (grid_w + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
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
            pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)

        return pos_embed.unsqueeze(0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        B, N_nomask = x.shape[:2]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - (N_nomask - 1), 1)       # [B, N_mask, C], N_mask = (N-1) - N_nomask
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)                                       # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed w/ cls token
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.transformer:
            x = block(x)
        x = self.decoder_norm(x)

        # predict pixels
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


# ------------------------ MAE Vision Transformer ------------------------
## Masked ViT
class MAE_VisionTransformer(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 784,
                 de_emb_dim    :int   = 512,
                 en_num_layers :int   = 12,
                 de_num_layers :int   = 12,
                 en_num_heads  :int   = 12,
                 de_num_heads  :int   = 16,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 mask_ratio    :float = 0.75,
                 is_train      :bool  = False,
                 norm_pix_loss :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.is_train = is_train
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        ## encoder
        self.en_emb_dim = en_emb_dim
        self.en_num_layers = en_num_layers
        self.en_num_heads = en_num_heads
        self.mask_ratio = mask_ratio
        ## decoder
        self.de_emb_dim = de_emb_dim
        self.de_num_layers = de_num_layers
        self.de_num_heads = de_num_heads
        self.norm_pix_loss = norm_pix_loss
        # -------- network parameters --------
        self.mae_encoder = MAE_ViT_Encoder(
            img_size, patch_size, img_dim, en_emb_dim, en_num_layers, en_num_heads, qkv_bias, mlp_ratio, dropout, mask_ratio)
        self.mae_decoder = MAE_ViT_Decoder(
            img_size, patch_size, img_dim, en_emb_dim, de_emb_dim, de_num_layers, de_num_heads, qkv_bias, mlp_ratio, dropout, norm_pix_loss)

    def patchify(self, imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x, patch_size):
        """
        x: (B, N, patch_size**2 *3)
        imgs: (B, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def compute_loss(self, x, output):
        """
        imgs: [B, 3, H, W]
        pred: [B, N, C], C = p*p*3
        mask: [B, N], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        pred, mask = output["x_pred"], output["mask"]
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss

    def forward(self, x):
        imgs = x
        x, mask, ids_restore = self.mae_encoder(x)
        x = self.mae_decoder(x, ids_restore)
        output = {
            'x_pred': x,
            'mask': mask
        }

        if self.is_train:
            loss = self.compute_loss(imgs, output)
            output["loss"] = loss

        return output


# ------------------------ Model Functions ------------------------
def mae_vit_nano(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, is_train=False, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 192,
                                  de_emb_dim    = 512,
                                  en_num_layers = 12,
                                  de_num_layers = 8,
                                  en_num_heads  = 12,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  is_train      = is_train,
                                  norm_pix_loss = norm_pix_loss)

    return model

def mae_vit_tiny(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, is_train=False, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 384,
                                  de_emb_dim    = 512,
                                  en_num_layers = 12,
                                  de_num_layers = 8,
                                  en_num_heads  = 12,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  is_train      = is_train,
                                  norm_pix_loss = norm_pix_loss)

    return model

def mae_vit_base(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, is_train=False, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 768,
                                  de_emb_dim    = 512,
                                  en_num_layers = 12,
                                  de_num_layers = 8,
                                  en_num_heads  = 12,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  is_train      = is_train,
                                  norm_pix_loss = norm_pix_loss)

    return model

def mae_vit_large(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, is_train=False, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 1024,
                                  de_emb_dim    = 512,
                                  en_num_layers = 24,
                                  de_num_layers = 8,
                                  en_num_heads  = 16,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  is_train      = is_train,
                                  norm_pix_loss = norm_pix_loss)

    return model

def mae_vit_huge(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, is_train=False, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 1280,
                                  de_emb_dim    = 512,
                                  en_num_layers = 32,
                                  de_num_layers = 8,
                                  en_num_heads  = 16,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  is_train      = is_train,
                                  norm_pix_loss = norm_pix_loss)

    return model


if __name__ == '__main__':
    import torch
    from thop import profile

    # build model
    bs, c, h, w = 2, 3, 224, 224
    is_train = True
    x = torch.randn(bs, c, h, w)
    model = mae_vit_tiny(patch_size=16, is_train=is_train)

    # inference
    outputs = model(x)
    if "loss" in outputs:
        print("Loss: ", outputs["loss"].item())

    # compute FLOPs & Params
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

    