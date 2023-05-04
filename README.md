# Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=learning-3d-representations-from-2d-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=learning-3d-representations-from-2d-pre)

Official implementation of ['Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders'](https://arxiv.org/pdf/2212.06785.pdf).

The paper has been accepted by **CVPR 2023** 🔥.

## News
* The pre-training and fine-tuning code of I2P-MAE has been released.
* The 3D-only variant of I2P-MAE is our previous work, [Point-M2AE](https://arxiv.org/pdf/2205.14401.pdf), accepted by **NeurIPS 2022** and [open-sourced](https://github.com/ZrrSkywalker/Point-M2AE). We have released its pre-training and fine-tuning code.
* 📣 Please check our latest work [Point-NN, Parameter is Not All You Need](https://github.com/ZrrSkywalker/Point-NN) accepted by **CVPR 2023**, which, for the first time, acheives 3D understanding with $\color{darkorange}{No\ Parameter\ or\ Training\.}$ 💥
* 📣 Please check our latest work [PiMAE](https://github.com/BLVLab/PiMAE) accepted by **CVPR 2023**, which promotes 3D and 2D interaction to improve 3D object detection performance.

## Introduction

Comparison with existing MAE-based 3D models on the three spilts of ScanObjectNN:
| Method | Parameters | GFlops| Extra Data | OBJ-BG | OBJ-ONLY| PB-T50-RS|
| :-----: | :-----: |:-----:| :-----: | :-----:| :-----:| :-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M |4.8| -|87.43% |88.12% |83.07 %| 
| [ACT](https://github.com/RunpeiDong/ACT) | 22.1M |4.8| 2D|92.48%| 91.57% | 87.88% | 
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | 22.1M |4.8| -|90.02%|88.29%|85.18%|
| [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE)| 12.9M |3.6| -|91.22%|88.81%|86.43%|
| **I2P-MAE** | **12.9M** |**3.6**| **2D**|**94.15%**|**91.57%**|**90.11%**|

We propose an alternative to obtain superior 3D representations from 2D pre-trained models via **I**mage-to-**P**oint Masked Autoencoders, named as **I2P-MAE**. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we conduct two types of image-to-point learning schemes: 2D-guided masking and 2D-semantic reconstruction. In this way, the 3D network can effectively inherit high-level 2D semantics learned from rich image data for discriminative 3D modeling.

<div align="center">
  <img src="pipeline.png"/>
</div>

## I2P-MAE Models

### Pre-training
Guided by pre-trained CLIP on ShapeNet, I2P-MAE is evaluated by **Linear SVM** on ModelNet40 and ScanObjectNN (OBJ-BG split) datasets, without downstream fine-tuning:
| Task | Dataset | Config | MN40 Acc.| OBJ-BG Acc.| Ckpts | Logs |   
| :-----: | :-----: |:-----:| :-----: | :-----:| :-----:|:-----:|
| Pre-training | ShapeNet |[i2p-mae.yaml](./cfgs/pre-training/i2p-mae.yaml)| 93.35% | 87.09% | [pre-train.pth](https://drive.google.com/file/d/1TYKHdLwu9DKLgsnvsY4fpgpNowCHErFZ/view?usp=share_link) | [log](https://drive.google.com/file/d/11kkgTQoUJVLYKk1Xbo0XtQPCqh50my-G/view?usp=share_link) |

### Fine-tuning
Synthetic shape classification on ModelNet40 with 1k points:
| Task  | Config | Acc.| Vote| Ckpts | Logs |   
| :-----: | :-----:| :-----:| :-----: | :-----:|:-----:|
| Classification | [modelnet40.yaml]()|93.67%| 94.06% | [modelnet40.pth]() | [modelnet40.log]() |

Real-world shape classification on ScanObjectNN:
| Task | Split | Config | Acc.| Ckpts | Logs |   
| :-----: | :-----:|:-----:| :-----:| :-----:|:-----:|
| Classification | PB-T50-RS|[scan_pb.yaml]() | 90.11%| [scan_pd.pth]() | [scan_pd.log]() |
| Classification |OBJ-BG| [scan_obj-bg.yaml]() | 94.15%| - | - |
| Classification | OBJ-ONLY| [scan_obj.yaml]() | 91.57%| - | - |


## Requirements

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/ZrrSkywalker/I2P-MAE.git
cd I2P-MAE

conda create -n i2pmae python=3.7
conda activate i2pmae

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
# e.g., conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Datasets
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially for Linear SVM evaluation, download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.

The final directory structure should be:
```
│I2P-MAE/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──modelnet40_ply_hdf5_2048/  # Specially for Linear SVM
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──...
```

## Get Started

### Pre-training
I2P-MAE is pre-trained on ShapeNet dataset with the config file `cfgs/pre-training/i2p-mae.yaml`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/i2p-mae.yaml --exp_name pre-train
```

To evaluate the pre-trained I2P-MAE by **Linear SVM**, create a folder `ckpts/` and download the [pre-train.pth]() into it. Use the configs in `cfgs/linear-svm/` and indicate the evaluation dataset by `--test_svm`.

For ModelNet40, run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/linear-svm/modelnet40.yaml --test_svm modelnet40 --exp_name test_svm --ckpts ./ckpts/pre-train.pth
```
For ScanObjectNN (OBJ-BG split), run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/linear-svm/scan_obj-bg.yaml --test_svm scan --exp_name test_svm --ckpts ./ckpts/pre-train.pth
```

### Fine-tuning
Please create a folder `ckpts/` and download the [pre-train.pth]() into it. The fine-tuning configs are in `cfgs/fine-tuning/`.

For ModelNet40, run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/modelnet40.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```

For the three splits of ScanObjectNN, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_pb.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_obj.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_obj-bg.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```


## Acknowledgement
This repo benefits from [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), and [CLIP](https://github.com/openai/CLIP). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
