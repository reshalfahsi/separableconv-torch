# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import warnings
import torch.nn as nn

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from typing import Callable, Union
from separableconv.nn.base import _SeparableConv


# Mostly modified from:
#     - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
#     - https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py


class SeparableConv1d(_SeparableConv):
    r"""Applies a 1D depthwise separable convolution over an input signal composed of several input
    planes as described in the paper
    `Xception: Deep Learning with Depthwise Separable Convolutions <https://arxiv.org/abs/1610.02357>`__ .

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        normalization_dw (str, optional): depthwise convolution normalization. Default: 'bn'
        normalization_pw (str): pointwise convolution normalization. Default: 'bn'
        activation_dw (Callable[..., torch.nn.Module], optional): depthwise convolution activation. Default: ``torch.nn.ReLU``
        activation_pw (Callable[..., torch.nn.Module], optional): pointwise convolution activation. Default: ``torch.nn.ReLU``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        padding_mode: str = "zeros",
        dilation: _size_1_t = 1,
        depth_multiplier: int = 1,
        normalization_dw: str = "bn",
        normalization_pw: str = "bn",
        activation_dw: Callable[..., nn.Module] = nn.ReLU,
        activation_pw: Callable[..., nn.Module] = nn.ReLU,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(SeparableConv1d, self).__init__()

        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)

        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv1d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.BatchNorm1d(expansion_channels)
            if normalization_dw == "bn"
            else nn.InstanceNorm1d(expansion_channels)
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm1d`` or 'in' for ``nn.InstanceNorm1d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv1d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.pwconv_normalization = (
            nn.BatchNorm1d(out_channels)
            if normalization_pw == "bn"
            else nn.InstanceNorm1d(out_channels)
            if normalization_pw == "in"
            else None
        )

        if self.pwconv_normalization is None:
            warnings.warn(
                "normalization_pw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm1d`` or 'in' for ``nn.InstanceNorm1d``."
            )

        self.pwconv_activation = activation_pw()


class SeparableConv2d(_SeparableConv):
    r"""Applies a 2D depthwise separable convolution over an input signal composed of several input
    planes as described in the paper
    `Xception: Deep Learning with Depthwise Separable Convolutions <https://arxiv.org/abs/1610.02357>`__ .

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        normalization_dw (str, optional): depthwise convolution normalization. Default: 'bn'
        normalization_pw (str): pointwise convolution normalization. Default: 'bn'
        activation_dw (Callable[..., torch.nn.Module], optional): depthwise convolution activation. Default: ``torch.nn.ReLU``
        activation_pw (Callable[..., torch.nn.Module], optional): pointwise convolution activation. Default: ``torch.nn.ReLU``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        padding_mode: str = "zeros",
        dilation: _size_2_t = 1,
        depth_multiplier: int = 1,
        normalization_dw: str = "bn",
        normalization_pw: str = "bn",
        activation_dw: Callable[..., nn.Module] = nn.ReLU,
        activation_pw: Callable[..., nn.Module] = nn.ReLU,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(SeparableConv2d, self).__init__()

        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)

        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv2d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.BatchNorm2d(expansion_channels)
            if normalization_dw == "bn"
            else nn.InstanceNorm2d(expansion_channels)
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm2d`` or 'in' for ``nn.InstanceNorm2d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv2d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.pwconv_normalization = (
            nn.BatchNorm2d(out_channels)
            if normalization_pw == "bn"
            else nn.InstanceNorm2d(out_channels)
            if normalization_pw == "in"
            else None
        )

        if self.pwconv_normalization is None:
            warnings.warn(
                "normalization_pw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm2d`` or 'in' for ``nn.InstanceNorm2d``."
            )

        self.pwconv_activation = activation_pw()


class SeparableConv3d(_SeparableConv):
    r"""Applies a 3D depthwise separable convolution over an input signal composed of several input
    planes as described in the paper
    `Xception: Deep Learning with Depthwise Separable Convolutions <https://arxiv.org/abs/1610.02357>`__ .

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        normalization_dw (str, optional): depthwise convolution normalization. Default: 'bn'
        normalization_pw (str): pointwise convolution normalization. Default: 'bn'
        activation_dw (Callable[..., torch.nn.Module], optional): depthwise convolution activation. Default: ``torch.nn.ReLU``
        activation_pw (Callable[..., torch.nn.Module], optional): pointwise convolution activation. Default: ``torch.nn.ReLU``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        padding_mode: str = "zeros",
        dilation: _size_3_t = 1,
        depth_multiplier: int = 1,
        normalization_dw: str = "bn",
        normalization_pw: str = "bn",
        activation_dw: Callable[..., nn.Module] = nn.ReLU,
        activation_pw: Callable[..., nn.Module] = nn.ReLU,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(SeparableConv3d, self).__init__()

        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)

        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv3d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.BatchNorm3d(expansion_channels)
            if normalization_dw == "bn"
            else nn.InstanceNorm3d(expansion_channels)
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm3d`` or 'in' for ``nn.InstanceNorm3d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv3d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.pwconv_normalization = (
            nn.BatchNorm3d(out_channels)
            if normalization_pw == "bn"
            else nn.InstanceNorm3d(out_channels)
            if normalization_pw == "in"
            else None
        )

        if self.pwconv_normalization is None:
            warnings.warn(
                "normalization_pw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm3d`` or 'in' for ``nn.InstanceNorm3d``."
            )

        self.pwconv_activation = activation_pw()
