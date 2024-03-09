import torch


# Basic Warmup Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor

    def set_lr(self, optimizer, cur_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / self.base_lr
            param_group['lr'] = cur_lr * ratio

    def __call__(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr)


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "step":
        lr_step = [args.max_epoch // 3, args.max_epoch // 3 * 2]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - args.wp_epoch - 1, eta_min=args.min_lr)
    else:
        raise NotImplementedError("Unknown lr scheduler: {}".format(args.lr_scheduler))
    
    print("=================== LR Scheduler information ===================")
    print("LR Scheduler: ", args.lr_scheduler)

    if args.resume and args.resume != 'None':
        print('- Load lr scheduler from the checkpoint: ', args.resume)
        checkpoint = torch.load(args.resume)
        scheduler.load_state_dict(checkpoint.pop("lr_scheduler"))

    return scheduler
        