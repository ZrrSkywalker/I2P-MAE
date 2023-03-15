# Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=learning-3d-representations-from-2d-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=learning-3d-representations-from-2d-pre)

Official implementation of ['Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders'](https://arxiv.org/pdf/2212.06785.pdf).

The paper has been accepted by **CVPR 2023** 🔥.

## News
* 📣 Please check our latest work [Point-NN, Parameter is Not All You Need](https://github.com/ZrrSkywalker/Point-NN) accepted by **CVPR 2023**, which, for the first time, acheives 3D understanding with $\color{darkorange}{No\ Parameter\ or\ Training\.}$ 💥
* 📣 Please check our latest work [PiMAE](https://github.com/BLVLab/PiMAE) accepted by **CVPR 2023**, which promotes 3D and 2D interaction to improve 3D object detection performance.
* The 3D-only variant of I2P-MAE is our previous work, [Point-M2AE](https://arxiv.org/pdf/2205.14401.pdf), accepted by **NeurIPS 2022** and [open-sourced](https://github.com/ZrrSkywalker/Point-M2AE).

## Introduction
We propose an alternative to obtain superior 3D representations from 2D pre-trained models via **I**mage-to-**P**oint Masked Autoencoders, named as **I2P-MAE**. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we conduct two types of image-to-point learning schemes: 2D-guided masking and 2D-semantic reconstruction. In this way, the 3D network can effectively inherit high-level 2D semantics learned from rich image data for discriminative 3D modeling.

<div align="center">
  <img src="pipeline.png"/>
</div>

## Code
Comming soon.

## Acknowledgement
This repo benefits from [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), and [CLIP](https://github.com/openai/CLIP). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
