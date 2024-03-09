from copy import deepcopy
import os
import time
import math
import argparse
import datetime

# ---------------- Torch compoments ----------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Dataset compoments ----------------
from data import build_dataset, build_dataloader

# ---------------- Model compoments ----------------
from models import build_model

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, load_model, save_model
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_scheduler import build_lr_scheduler, LinearWarmUpLrScheduler
from utils.com_flops_params import FLOPs_and_Params
from utils.lars import LARS

# ---------------- Training engine ----------------
from engine_finetune import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size.')    
    parser.add_argument('--img_dim', type=int, default=3,
                        help='3 for RGB; 1 for Gray.')    
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch_size.')    
    parser.add_argument('--color_format', type=str, default='rgb',
                        help='color format: rgb or bgr')    
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size on all GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=5, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=50, 
                        help='max epoch')
    parser.add_argument('--eval_epoch', type=int, default=5, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of classes.')
    # Model
    parser.add_argument('-m', '--model', type=str, default='vit_tiny',
                        help='model name')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='load pretrained weight.')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    parser.add_argument('--learnable_pos', action='store_true', default=False,
                        help='learnable position embedding.')
    parser.add_argument('--drop_path', type=float, default=0.,
                        help='drop_path')
    # Optimizer
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default='cosine',
                        help='step, cosine')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.1,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='the final lr')
    parser.add_argument('-accu', '--grad_accumulate', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    # DDP
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='the number of local rank.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    path_to_save = os.path.join(args.path_to_save, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    args.output_dir = path_to_save
    
    # ------------------------- Build DDP environment -------------------------
    ## LOCAL_RANK is the global GPU number tag, the value range is [0, world_size - 1].
    ## LOCAL_PROCESS_RANK is the number of the GPU of each machine, not global.
    local_rank = local_process_rank = -1
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    print_rank_0(args, local_rank)
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)

    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        if torch.cuda.is_available():
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            print('There is no available GPU.')
            args.cuda = False
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # ------------------------- Build Tensorboard -------------------------
    tblogger = None
    if local_rank <= 0 and args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, time_stamp)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, is_train=True)
    val_dataset   = build_dataset(args, is_train=False)

    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    val_dataloader   = build_dataloader(args, val_dataset,   is_train=False)

    print('=================== Dataset Information ===================')
    print('Train dataset size : ', len(train_dataset))
    print('Val dataset size   : ', len(val_dataset))

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    model.classifier = torch.nn.Sequential(
        nn.BatchNorm1d(model.classifier.in_features, affine=False, eps=1e-6),
        model.classifier)
    model.train().to(device)
    print(model)
    if local_rank <= 0:
        model_copy = deepcopy(model)
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, size=args.img_size)
        model_copy.train()
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------- Build Optimzier -------------------------
    ## learning rate
    args.base_lr = args.base_lr / 256 * args.batch_size * args.grad_accumulate    # auto scale lr
    optimizer = LARS(model_without_ddp.classifier.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    # ------------------------- Build Lr Scheduler -------------------------
    lr_scheduler_warmup = LinearWarmUpLrScheduler(args.base_lr, wp_iter=args.wp_epoch * len(train_dataloader))
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # ------------------------- Build Criterion -------------------------
    criterion = torch.nn.CrossEntropyLoss()
    load_model(args=args, model_without_ddp=model_without_ddp,
               optimizer=optimizer, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler)

    # ------------------------- Eval before Train Pipeline -------------------------
    if args.eval:
        print('evaluating ...')
        test_stats = evaluate(val_dataloader, model, device, local_rank)
        print('Eval Results: [loss: %.2f][acc1: %.2f][acc5 : %.2f]' %
                (test_stats['loss'], test_stats['acc1'], test_stats['acc5']), flush=True)
        exit(0)

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    max_accuracy = -1.0
    print_rank_0("=============== Start training for {} epochs ===============".format(args.max_epoch), local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        train_one_epoch(args, device, model, train_dataloader, optimizer, epoch,
                        lr_scheduler_warmup, loss_scaler, criterion, local_rank, tblogger)

        # LR scheduler
        if (epoch + 1) > args.wp_epoch:
            lr_scheduler.step()

        # Evaluate
        if (epoch % args.eval_epoch) == 0 or (epoch + 1 == args.max_epoch):
            test_stats = evaluate(val_dataloader, model, device, local_rank)
            print_rank_0(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%", local_rank)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print_rank_0(f'Max accuracy: {max_accuracy:.2f}%', local_rank)

            # Save model
            if local_rank <= 0:
                print('- saving the model after {} epochs ...'.format(epoch))
                save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                           optimizer=optimizer, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler, epoch=epoch, acc1=max_accuracy)
        if args.distributed:
            dist.barrier()

        if tblogger is not None:
            tblogger.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            tblogger.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            tblogger.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()