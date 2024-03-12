# ------------------- Model setting -------------------
MODEL=$1
BATCH_SIZE=$2
DATASET=$3
DATASET_PATH=$4
PRETRAINED_MODEL=$5
RESUME=$6

# ------------------- Training setting -------------------
OPTIMIZER="adamw"
LRSCHEDULER="cosine"
MIN_LR=1e-6
WEIGHT_DECAY=0.05

if [ $MODEL == "vit_h" ]; then
    MAX_EPOCH=50
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    LAYER_DECAY=0.75
    DROP_PATH=0.3
elif [ $MODEL == "vit_l" ]; then
    MAX_EPOCH=50
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    LAYER_DECAY=0.75
    DROP_PATH=0.2
else
    MAX_EPOCH=100
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.0005
    LAYER_DECAY=0.65
    DROP_PATH=0.1
fi

# ------------------- Dataset config -------------------
if [[ $DATASET == "cifar10" || $DATASET == "cifar100" ]]; then
    # Data root
    ROOT="none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    # Data root
    ROOT="path/to/imagenet"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
elif [[ $DATASET == "custom" ]]; then
    # Data root
    ROOT="path/to/custom"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
else
    echo "Unknown dataset!!"
    exit 1
fi


# ------------------- Training pipeline -------------------
WORLD_SIZE=1
if [ $WORLD_SIZE == 1 ]; then
    python main_finetune.py \
            --cuda \
            --root ${ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --lr_scheduler ${LRSCHEDULER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --layer_decay ${LAYER_DECAY} \
            --weight_decay ${WEIGHT_DECAY} \
            --reprob 0.25 \
            --mixup 0.8 \
            --cutmix 1.0 \
            --resume ${RESUME} \
            --pretrained ${PRETRAINED_MODEL}
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port 1668 main_finetune.py \
            --cuda \
            -dist \
            --root ${ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --lr_scheduler ${LRSCHEDULER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --layer_decay ${LAYER_DECAY} \
            --weight_decay ${WEIGHT_DECAY} \
            --reprob 0.25 \
            --mixup 0.8 \
            --cutmix 1.0 \
            --resume ${RESUME} \
            --pretrained ${PRETRAINED_MODEL}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi