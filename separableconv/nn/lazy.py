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

from torch import Tensor
from torch.nn.parameter import UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from typing import Callable, Union, Tuple
from separableconv.nn.base import _SeparableConv


# Mostly modified from:
#     - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
#     - https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py


class _LazyDepthwiseConvXdMixin(LazyModuleMixin):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter

    def reset_parameters(self) -> None:
        # has_uninitialized_params is defined in parent class and it is using a protocol on self
        if not self.has_uninitialized_params() and self.in_channels != 0:  # type: ignore[misc]
            # "type:ignore[..]" is required because mypy thinks that "reset_parameters" is undefined
            # in super class. Turns out that it is defined in _ConvND which is inherited by any class
            # that also inherits _LazyConvXdMixin
            super().reset_parameters()  # type: ignore[misc]

    # Signature of "initialize_parameters" is incompatible with the definition in supertype LazyModuleMixin
    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        # defined by parent class but using a protocol
        if self.has_uninitialized_params():  # type: ignore[misc]
            self.in_channels = self._get_in_channels(input)
            depth_multiplier = self.out_channels
            self.out_channels = max(
                self.in_channels * int(depth_multiplier), self.in_channels
            )
            if self.in_channels * depth_multiplier != self.out_channels:
                raise ValueError("depth_multiplier must be integer>=1")
            self.groups = self.in_channels
            assert isinstance(self.weight, UninitializedParameter)
            self.weight.materialize(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
            )
            if self.bias is not None:
                assert isinstance(self.bias, UninitializedParameter)
                self.bias.materialize((self.out_channels,))
            self.reset_parameters()

    # Function to extract in_channels from first input.
    def _get_in_channels(self, input: Tensor) -> int:
        num_spatial_dims = self._get_num_spatial_dims()
        num_dims_no_batch = num_spatial_dims + 1  # +1 for channels dim
        num_dims_batch = num_dims_no_batch + 1
        if input.dim() not in (num_dims_no_batch, num_dims_batch):
            raise RuntimeError(
                "Expected {}D (unbatched) or {}D (batched) input to {}, but "
                "got input of size: {}".format(
                    num_dims_no_batch,
                    num_dims_batch,
                    self.__class__.__name__,
                    input.shape,
                )
            )
        return input.shape[1] if input.dim() == num_dims_batch else input.shape[0]

    # Function to return the number of spatial dims expected for inputs to the module.
    # This is expected to be implemented by subclasses.
    def _get_num_spatial_dims(self) -> int:
        raise NotImplementedError()


class LazyDepthwiseConv1d(_LazyDepthwiseConvXdMixin, nn.Conv1d):  # type: ignore[misc]
    r"""A LazyDepthwiseConv1d module.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    # super class define this variable as None. "type: ignore[..] is required
    # since we are redefining the variable.
    cls_to_become = nn.Conv1d  # type: ignore[assignment]

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        depth_multiplier: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            1,
            # bias is hardcoded to False to avoid creating tensor
            # that will soon be overwritten.
            False,
            padding_mode,
            **factory_kwargs
        )
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = depth_multiplier
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 1


class LazyDepthwiseConv2d(_LazyDepthwiseConvXdMixin, nn.Conv2d):  # type: ignore[misc]
    r"""A LazyDepthwiseConv2d module.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    # super class define this variable as None. "type: ignore[..] is required
    # since we are redefining the variable.
    cls_to_become = nn.Conv2d  # type: ignore[assignment]

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        depth_multiplier: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            1,
            # bias is hardcoded to False to avoid creating tensor
            # that will soon be overwritten.
            False,
            padding_mode,
            **factory_kwargs
        )
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = depth_multiplier
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 2


class LazyDepthwiseConv3d(_LazyDepthwiseConvXdMixin, nn.Conv3d):  # type: ignore[misc]
    r"""A LazyDepthwiseConv3d module.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    # super class define this variable as None. "type: ignore[..] is required
    # since we are redefining the variable.
    cls_to_become = nn.Conv3d  # type: ignore[assignment]

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        depth_multiplier: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            1,
            # bias is hardcoded to False to avoid creating tensor
            # that will soon be overwritten.
            False,
            padding_mode,
            **factory_kwargs
        )
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = depth_multiplier
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 3


class LazySeparableConv1d(_SeparableConv):
    r"""A ``nn.SeparableConv1d`` module with lazy initialization of
    the ``in_channels`` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Args:
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
        super(LazySeparableConv1d, self).__init__()

        self.dwconv = LazyDepthwiseConv1d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depth_multiplier=depth_multiplier,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.LazyBatchNorm1d()
            if normalization_dw == "bn"
            else nn.LazyInstanceNorm1d()
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.LazyBatchNorm1d`` or 'in' for ``nn.LazyInstanceNorm1d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.LazyConv1d(
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


class LazySeparableConv2d(_SeparableConv):
    r"""A ``nn.SeparableConv2d`` module with lazy initialization of
    the ``in_channels`` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Args:
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
        super(LazySeparableConv2d, self).__init__()

        self.dwconv = LazyDepthwiseConv2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depth_multiplier=depth_multiplier,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.LazyBatchNorm2d()
            if normalization_dw == "bn"
            else nn.LazyInstanceNorm2d()
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.LazyBatchNorm2d`` or 'in' for ``nn.LazyInstanceNorm2d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.LazyConv2d(
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


class LazySeparableConv3d(_SeparableConv):
    r"""A ``nn.SeparableConv3d`` module with lazy initialization of
    the ``in_channels`` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Args:
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
        super(LazySeparableConv3d, self).__init__()

        self.dwconv = LazyDepthwiseConv3d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depth_multiplier=depth_multiplier,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.LazyBatchNorm3d()
            if normalization_dw == "bn"
            else nn.LazyInstanceNorm3d()
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.LazyBatchNorm3d`` or 'in' for ``nn.LazyInstanceNorm3d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.LazyConv3d(
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
