# SAMI: Masked AutoEncoders leveraging Segment-Anything
**Unofficial pytorch implementation of Masked AutoEncoder**

Based on my understanding of EfficientSAM's SAMI framework and the technical details given in the paper, I tried to implement the SAMI pre-training framework, using SAM's ViT to improve the performance of small-scale ViT models, including ViT-Tiny and ViT-Small.

Unfortunately, I currently do not have sufficient computing resources to verify whether my implementation can reproduce the SAMI experimental results in the EfficientSAM paper.

First of all, you need to follow the requirements of this [README](./checkpoints/README.md) file to prepare the SAM's ViT checkpoint, which will be used as the teacher
model to supervise the small-scale ViT in the SAMI pretraining stage.

## 1. Pretrain
We have kindly provided the bash script `train_pretrain.sh` file for pretraining. You can modify some hyperparameters in the script file according to your own needs.

- Single GPU

```Shell
# bash train_pretrain.sh <model> <teacher model> <batch size> <data> <data path> <world size> <resume>
bash train_pretrain.sh vit_t vit_h 256 imagenet_1k /path/to/imagenet_1k/ 1 None
```

- Multi GPUs

```Shell
# bash train_pretrain.sh <model> <teacher model> <batch size> <data> <data path> <world size> <resume>
bash train_pretrain.sh vit_t vit_h 256 imagenet_1k /path/to/imagenet_1k/ 8 None
```

## 2. Finetune
We have kindly provided the bash script `train_finetune.sh` file for finetuning. You can modify some hyperparameters in the script file according to your own needs.

- Single GPU

```Shell
# bash train_pretrain.sh <model> <batch size> <data> <data path> <world size> <resume>
bash train_finetune.sh vit_t 256 imagenet_1k /path/to/imagenet_1k/ 1 None
```

- Multi GPUs

```Shell
# bash train_pretrain.sh <model> <batch size> <data> <data path> <world size> <resume>
bash train_finetune.sh vit_t 256 imagenet_1k /path/to/imagenet_1k/ 8 None
```

## 3. Evaluate 
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_finetune.py --dataset cifar10 -m vit_t --batch_size 256 --img_size 32 --patch_size 2 --eval --resume path/to/checkpoint
```

- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_finetune.py --dataset imagenet_1k --root /path/to/imagenet_1k -m vit_t --batch_size 256 --img_size 224 --patch_size 16 --eval --resume path/to/checkpoint
```

## 4. Experiments

### Classification: ImageNet-1K
- We use the SAM's `ViT-H` as the teacher to supervise the small-scale ViT.

|  Method  |  Model  | Epoch | Top 1     | Weight |  MAE weight  |
|  :---:   | :---:   | :---: | :---:     | :---:  |    :---:     |
|   SAMI   |  ViT-T  | 100   |           |  |  |
|   SAMI   |  ViT-S  | 100   |           |  |  |

### Object detection: COCO
- We use the small ViT pretrained by the SAMI as the backbone of `ViTDet`.

|  Method  |  Model  | Backbone | Epoch |   Top 1   | Weight |  MAE weight  |
|  :---:   | :---:   |   :---:  | :---: |   :---:   | :---:  |    :---:     |
|   SAMI   |  ViTDet |   Vit-T  |  100  |           |  |  |
|   SAMI   |  ViTDet |   Vit-S  |  100  |           |  |  |



## 6. Acknowledgment
Thank you to **Kaiming He** for his inspiring work on [MAE](http://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf). His research effectively elucidates the semantic distinctions between vision and language, offering valuable insights for subsequent vision-related studies. I would also like to express my gratitude for the official source code of [MAE](https://github.com/facebookresearch/mae). Additionally, I appreciate the efforts of [**IcarusWizard**](https://github.com/IcarusWizard) for reproducing the [MAE](https://github.com/IcarusWizard/MAE) implementation.
