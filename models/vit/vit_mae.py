import math
import torch
import torch.nn as nn

try:
    from .common import ViTBlock, PatchEmbed
except:
    from  common import ViTBlock, PatchEmbed


# ------------------------ Basic Modules ------------------------
class MaeEncoder(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 patch_embed_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float,
                 act_layer: nn.GELU,
                 mask_ratio: float = 0.75,
                 dropout: float = 0.0,
                 ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        # ----------- Basic parameters -----------
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.patch_embed_dim = patch_embed_dim
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        # ----------- Model parameters -----------
        self.patch_embed = PatchEmbed(in_chans, patch_embed_dim, patch_size, 0, patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, patch_embed_dim), requires_grad=False)
        self.norm_layer  = nn.LayerNorm(patch_embed_dim)
        self.blocks      = nn.ModuleList([
            ViTBlock(patch_embed_dim, num_heads, mlp_ratio, True, act_layer=act_layer, dropout=dropout)
            for _ in range(depth)])
        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_posembed(self.pos_embed.shape[-1], int(self.num_patches**.5))
        self.pos_embed.data.copy_(pos_embed)

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

    def get_posembed(self, embed_dim, grid_size, temperature=10000):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embed
        x = self.patch_embed(x)
        # [B, C, H, W] -> [B, C, N] -> [B, N, C], N = H x W
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # add pos embed
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm_layer(x)
        
        return x, mask, ids_restore

class MaeDecoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 en_emb_dim    :int   = 784,
                 out_dim       :int   = 1024,
                 de_emb_dim    :int   = 512,
                 de_num_layers :int   = 12,
                 de_num_heads  :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 norm_pix_loss :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.out_dim = out_dim
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
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, de_emb_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_norm      = nn.LayerNorm(de_emb_dim)
        self.decoder_pred      = nn.Linear(de_emb_dim, out_dim, bias=True)
        self.blocks            = nn.ModuleList([
            ViTBlock(de_emb_dim, de_num_heads, mlp_ratio, qkv_bias, dropout=dropout)
            for _ in range(de_num_layers)])
        
        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = self.get_posembed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5))
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

    def get_posembed(self, embed_dim, grid_size, temperature=10000):
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

        return pos_embed.unsqueeze(0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        B, N_nomask = x.shape[:2]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - (N_nomask - 1), 1)     # [B, N_mask, C], N_mask = (N-1) - N_nomask
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed w/ cls token
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # predict pixels
        x = self.decoder_pred(x)

        return x


# ------------------------ MAE Vision Transformer ------------------------
class ViTforMaskedAutoEncoder(nn.Module):
    def __init__(self,
                 encoder :MaeEncoder,
                 decoder :MaeDecoder,
                 norm_pix_loss :bool = False):
        super().__init__()
        self.mae_encoder = encoder
        self.mae_decoder = decoder
        self.norm_pix_loss = norm_pix_loss

    def compute_loss(self, output, target):
        """
        pred: [B, N, C_ot]
        mask: [B, N], 0 is keep, 1 is remove, 
        """
        pred = output["x_pred"]
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        loss = loss.mean()
        
        return loss

    def forward(self, x, target=None):
        x, mask, ids_restore = self.mae_encoder(x)
        x = self.mae_decoder(x, ids_restore)
        output = {
            'x_pred': x,
            'mask': mask
        }

        if self.training:
            assert target is not None
            loss = self.compute_loss(output, target)
            output["loss"] = loss

        return output


# ------------------------ Model Functions ------------------------
def build_vit_mae(model_name="vit_t", img_size=224, patch_size=16, img_dim=3, out_dim=256, mask_ratio=0.75, norm_pix_loss=False):
    # ---------------- MAE Encoder ----------------
    if model_name == "vit_t":
        encoder = MaeEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=img_dim,
                             patch_embed_dim=192,
                             depth=12,
                             num_heads=3,
                             mlp_ratio=4.0,
                             act_layer=nn.GELU,
                             mask_ratio=mask_ratio,
                             dropout = 0.1)
    if model_name == "vit_s":
        encoder = MaeEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=img_dim,
                             patch_embed_dim=384,
                             depth=12,
                             num_heads=6,
                             mlp_ratio=4.0,
                             act_layer=nn.GELU,
                             mask_ratio=mask_ratio,
                             dropout = 0.1)
    if model_name == "vit_b":
        encoder = MaeEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=img_dim,
                             patch_embed_dim=768,
                             depth=12,
                             num_heads=12,
                             mlp_ratio=4.0,
                             act_layer=nn.GELU,
                             mask_ratio=mask_ratio,
                             dropout = 0.1)
    if model_name == "vit_l":
        encoder = MaeEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=img_dim,
                             patch_embed_dim=1024,
                             depth=24,
                             num_heads=16,
                             mlp_ratio=4.0,
                             act_layer=nn.GELU,
                             mask_ratio=mask_ratio,
                             dropout = 0.1)
    if model_name == "vit_h":
        encoder = MaeEncoder(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=img_dim,
                             patch_embed_dim=1280,
                             depth=32,
                             num_heads=16,
                             mlp_ratio=4.0,
                             act_layer=nn.GELU,
                             mask_ratio=mask_ratio,
                             dropout = 0.1)
    
    # ---------------- MAE Decoder ----------------
    decoder = MaeDecoder(img_size=img_size,
                         patch_size=patch_size,
                         en_emb_dim=encoder.patch_embed_dim,
                         out_dim=out_dim,
                         de_emb_dim=512,
                         de_num_layers=8,
                         de_num_heads=16,
                         qkv_bias=True,
                         mlp_ratio=4.0,
                         dropout=0.1,)
    
    return ViTforMaskedAutoEncoder(encoder, decoder, norm_pix_loss)


if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, c, h, w = 2, 3, 224, 224
    x = torch.randn(bs, c, h, w)
    # Prepare an feature map as the target
    patch_size = 16
    out_dim = 1024
    target = torch.randn(bs, h//patch_size*w//patch_size, out_dim)

    # Build model
    model = build_vit_mae(patch_size=patch_size, out_dim=out_dim)

    # Inference
    outputs = model(x, target)
    if "loss" in outputs:
        print("Loss: ", outputs["loss"].item())

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

