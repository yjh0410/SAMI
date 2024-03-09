import torch.utils.data as data

from .cifar import CifarDataset
from .imagenet import ImageNet1KDataset
from .custom import CustomDataset


def build_dataset(args, transform=None, is_train=False):
    # ----------------- CIFAR dataset -----------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        return CifarDataset(is_train, transform)
    
    # ----------------- ImageNet dataset -----------------
    elif args.dataset == 'imagenet_1k':
        args.num_classes = 1000
        return ImageNet1KDataset(args, is_train, transform)
    
    # ----------------- Customed dataset -----------------
    elif args.dataset == 'custom':
        assert args.num_classes is not None and isinstance(args.num_classes, int)
        return CustomDataset(args, is_train, transform)
    
    else:
        print("Unknown dataset: {}".format(args.dataset))
    

def build_dataloader(args, dataset, is_train=False):
    if is_train:
        sampler = data.distributed.DistributedSampler(dataset) if args.distributed else data.RandomSampler(dataset)
        per_gpu_batch = args.batch_size // args.world_size
        batch_sampler_train = data.BatchSampler(sampler, per_gpu_batch, drop_last=True if is_train else False)
        dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    else:
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader
