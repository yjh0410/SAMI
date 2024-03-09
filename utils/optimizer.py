import torch


def build_optimizer(model, base_lr, weight_decay):
    # ------------- Divide model's parameters -------------
    param_dicts = [], [], []
    norm_names = ["norm"] + ["norm{}".format(i) for i in range(10000)]
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[0].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[1].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[2].append(p)  # weight decay for all Non-NormLayers' weight

    # Build optimizer
    optimizer = torch.optim.AdamW(param_dicts[0], lr=base_lr, weight_decay=0.0)
    
    # Add param groups
    optimizer.add_param_group({"params": param_dicts[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[2], "weight_decay": weight_decay})
                                                        
    return optimizer
