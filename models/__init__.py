import os
import torch

from .vit import build_vit, build_vit_mae, ViTForImageClassification
from .vit_sam import build_vit_sam


def build_model(args, model_type='default'):
    assert args.model in ['vit_t', 'vit_s', 'vit_b', 'vit_l', 'vit_h'], "Unknown vit model: {}".format(args.model)
    if model_type == 'default':
        return build_vit(args.model, args.img_size, args.patch_size, args.img_dim)
    
    elif model_type == 'mae':
        return build_vit_mae(args.model, args.img_size, args.patch_size, args.img_dim, args.out_dim, args.mask_ratio, args.norm_pix_loss)
    
    elif model_type == 'cls':
        # Build image encoder
        image_encoder = build_vit(args.model, args.img_size, args.patch_size, args.img_dim)

        # Build classifier
        model = ViTForImageClassification(image_encoder, num_classes=args.num_classes, qkv_bias=True)

        # Load MAE pretrained
        if args.pretrained is not None:
            # check path
            if not os.path.exists(args.pretrained):
                print("No pretrained model.")
                return model
            print('- Loading pretrained from: {}'.format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            encoder_state_dict = checkpoint.pop("encoder")

            # load encoder weight into ViT's encoder
            model.encoder.load_state_dict(encoder_state_dict)

        return model
    
    else:
        raise NotImplementedError("Unknown model type: {}".format(model_type))
    
def build_sam_teacher(args):
    # We use the ViT of SAM as the teacher in SAMI pretraining.
    if args.teacher is None or args.teacher.lower() == "none":
        raise NotImplementedError("You should build a teacher model required by the SAMI.")
    else: 
        teacher = build_vit_sam(args.teacher, args.img_size, args.patch_size, args.img_dim, args.checkpoint)
        return teacher.eval()
