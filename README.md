# self-supervised-timm
Adapt TIMM backbone to your custom dataset with self-supervised learning and convert model into ONNX format for deployment.

## Why?
TIMM has a wide range of pre-trained models that can be used for transfer learning. However, the pre-trained models are trained on ImageNet dataset which may not be suitable for your custom dataset. 

This repository provides a way to adapt TIMM backbone to your custom dataset with self-supervised learning and convert the model into ONNX format for deployment.

## How?
Using self-supervised learning (SSL) we adapt the TIMM backbone to the custom dataset without any labels. Once the SSL training is done, we can optionally fine-tune the model with labels. Finally, we convert the model into ONNX format for deployment.

## Dataset

MVTec AD Zipper Images - https://www.kaggle.com/datasets/atthaariq/resized-zipper

## Setting Up
Make sure you have PyTorch installed. Next, install the required packages.

```bash
pip install fastai timm onnx self-supervised
```