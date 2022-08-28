<h1 align="center">
  Unofficial PyTorch Module - Depthwise Separable Convolution
</h1>


<div align="center">
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_10.30.20_PM.png" width="400">

An illustration of Depthwise Separable Convolution. Credit: [Depthwise Convolution Is All You Need for Learning Multiple Visual Domains](https://paperswithcode.com/paper/depthwise-convolution-is-all-you-need-for).
</div>


<div align="center">
  <a href="https://pepy.tech/project/separableconv-torch"><img src="https://pepy.tech/badge/separableconv-torch" alt="total downloads"></a>
  <a href="https://pepy.tech/project/separableconv-torch"><img src="https://pepy.tech/badge/separableconv-torch/month" alt="monthly downloads"></a>
  <a href="https://github.com/reshalfahsi/separableconv-torch/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"></a>
  <a href="https://badge.fury.io/py/separableconv-torch"><img src="https://badge.fury.io/py/separableconv-torch.svg" alt="pypi version"></a>
  <a href="https://github.com/reshalfahsi/separableconv-torch/actions/workflows/ci.yml"><img src="https://github.com/reshalfahsi/separableconv-torch/actions/workflows/ci.yml/badge.svg" alt="ci testing"></a>
  <a href="https://github.com/reshalfahsi/separableconv-torch/actions/workflows/package.yml"><img src="https://github.com/reshalfahsi/separableconv-torch/actions/workflows/package.yml/badge.svg" alt="package testing"></a>
</div>

PyTorch (unofficial) implementation of Depthwise Separable Convolution. This type of convolution is introduced by Chollet in [Xception: Deep Learning With Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357). This package provides ``SeparableConv1d``, ``SeparableConv2d``, ``SeparableConv3d``, ``LazySeparableConv1d``, ``LazySeparableConv2d``, and ``LazySeparableConv3d``. 


## Installation

Install `separableconv-torch` using `pip` (require: Python >=3.7).

```console
pip install separableconv-torch
```


## Parameters

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of channels in the input image | int |
| out_channels | Number of channels produced by the convolution | int |
| kernel_size | Size of the convolving kernel | int or tuple |
| stride | Stride of the convolution. Default: 1 | int or tuple, optional |
| padding | Padding added to all four sides of the input. Default: 0 | int, tuple or str, optional |
| padding_mode | ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'`` | string, optional|
| dilation | Spacing between kernel elements. Default: 1 | int or tuple, optional |
| depth_multiplier | The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to `in_channels * depth_multiplier`. Default: 1| int, optional |
| normalization_dw | depthwise convolution normalization. Default: 'bn' | str, optional |
| normalization_pw | pointwise convolution normalization. Default: 'bn' | str, optional |
| activation_dw | depthwise convolution activation. Default: ``torch.nn.ReLU`` | Callable[`...`, `torch.nn.Module`], optional |
| activation_pw | pointwise convolution activation. Default: ``torch.nn.ReLU`` | Callable[`...`, `torch.nn.Module`], optional |
| bias | If ``True``, adds a learnable bias to the output. Default: ``True`` | bool, optional |


## Example Usage

<details open>
<summary>For 1-dimensional case.</summary>


```python
import torch
import separableconv.nn as nn

# set input
input = torch.randn(4, 10, 100)

# define model
m = nn.SeparableConv1d(10, 30, 3)

# process input through model
output = m(input)
```
</details>


<details closed>
<summary>For 2-dimensional case.</summary>


```python
import torch
import separableconv.nn as nn

# set input
input = torch.randn(4, 10, 100, 100)

# define model
m = nn.SeparableConv2d(10, 30, 3)

# process input through model
output = m(input)
```
</details>



<details closed>
<summary>For 3-dimensional case.</summary>


```python
import torch
import separableconv.nn as nn

# set input
input = torch.randn(4, 10, 100, 100, 100)

# define model
m = nn.SeparableConv3d(10, 30, 3)

# process input through model
output = m(input)
```
</details>


<details closed>
<summary>Stacked SeparableConv2d.</summary>


```python
import torch
import separableconv.nn as nn

# set input
input = torch.randn(4, 3, 100, 100)

# define model
m = nn.Sequential(
        nn.SeparableConv2d(3, 32, 3),
        nn.SeparableConv2d(32, 64, 3),
        nn.SeparableConv2d(64, 96, 3))

# process input through model
output = m(input)
```
</details>


<details closed>
<summary>For lazy 2-dimensional case.</summary>


```python
import torch
import separableconv.nn as nn

# set input
input = torch.randn(4, 10, 100, 100)

# define model
m = nn.LazySeparableConv2d(30, 3)

# process input through model
output = m(input)
```
</details>
