import torch

from .vit.build import build_vit, build_mae_vit


def build_model(args, is_train=False):
    # --------------------------- ViT series ---------------------------
    if   args.model in ['vit_nano', 'vit_tiny', 'vit_base', 'vit_large', 'vit_huge']:
        model = build_vit(args)

    elif args.model in ['mae_vit_nano', 'mae_vit_tiny', 'mae_vit_base', 'mae_vit_large', 'mae_vit_huge']:
        model = build_mae_vit(args, is_train)

    if args.resume and args.resume.lower() != 'none':
        print('loading trained weight for <{}> from <{}>: '.format(args.model, args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    return model


def build_teacher(args):
    """We use the ViT encoder of SAM as the teacher."""
    return
