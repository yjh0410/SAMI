import os
import torch
from timm.models.layers import trunc_normal_

from .pos_embed import interpolate_pos_embed


# ------------------------ Vision Transformer ------------------------
from .vit import vit_nano, vit_tiny, vit_base, vit_large, vit_huge

def build_vit(args):
    # build vit model
    if args.model == 'vit_nano':
        model = vit_nano(args.img_size, args.patch_size, args.img_dim, args.num_classes, args.drop_path, args.learnable_pos)
    elif args.model == 'vit_tiny':
        model = vit_tiny(args.img_size, args.patch_size, args.img_dim, args.num_classes, args.drop_path, args.learnable_pos)
    elif args.model == 'vit_base':
        model = vit_base(args.img_size, args.patch_size, args.img_dim, args.num_classes, args.drop_path, args.learnable_pos)
    elif args.model == 'vit_large':
        model = vit_large(args.img_size, args.patch_size, args.img_dim, args.num_classes, args.drop_path, args.learnable_pos)
    elif args.model == 'vit_huge':
        model = vit_huge(args.img_size, args.patch_size, args.img_dim, args.num_classes, args.drop_path, args.learnable_pos)
    
    # load pretrained
    if args.pretrained is not None:
        # check path
        if not os.path.exists(args.pretrained):
            print("No pretrained model.")
            return model
        ## load mae pretrained model
        print('Loading pretrained from <{}> for <{}> ...'.format('mae_'+args.model, args.model))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # collect MAE-ViT's encoder weight
        encoder_state_dict = {}
        for k in list(checkpoint_state_dict.keys()):
            if 'mae_encoder' in k and k[12:] in model_state_dict.keys():
                encoder_state_dict[k[12:]] = checkpoint_state_dict[k]

        # interpolate position embedding
        interpolate_pos_embed(model, encoder_state_dict)

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict, strict=False)

        # manually initialize fc layer
        trunc_normal_(model.classifier.weight, std=2e-5)

    return model


# ------------------------ MAE Vision Transformer ------------------------
from .vit_mae import mae_vit_nano, mae_vit_tiny, mae_vit_base, mae_vit_large, mae_vit_huge

def build_mae_vit(args, is_train=False):
    # build vit model
    if args.model == 'mae_vit_nano':
        model = mae_vit_nano(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, is_train, args.norm_pix_loss)
    elif args.model == 'mae_vit_tiny':
        model = mae_vit_tiny(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, is_train, args.norm_pix_loss)
    elif args.model == 'mae_vit_base':
        model = mae_vit_base(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, is_train, args.norm_pix_loss)
    elif args.model == 'mae_vit_large':
        model = mae_vit_large(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, is_train, args.norm_pix_loss)
    elif args.model == 'mae_vit_huge':
        model = mae_vit_huge(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, is_train, args.norm_pix_loss)

    return model