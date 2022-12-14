# Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=learning-3d-representations-from-2d-pre)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=learning-3d-representations-from-2d-pre)

Official implementation of 'Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders'

## Introduction
We propose an alternative to obtain superior 3D representations from 2D pre-trained models via **I**mage-to-**P**oint Masked Autoencoders, named as **I2P-MAE**. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we conduct two types of image-to-point learning schemes: 2D-guided masking and 2D-semantic reconstruction. In this way, the 3D network can effectively inherit high-level 2D semantics learned from rich image data for discriminative 3D modeling.

<div align="center">
  <img src="pipeline.png"/>
</div>

## Code
Comming soon.

## Acknowledgement
This repo benefits from [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE) and [CLIP](https://github.com/openai/CLIP). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
