# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Maximum padded input width per convolution call on the NPU.
# CANN Conv2DV2 rejects widths above 1,000,000 in the tested environment.
_NPU_CONV1D_CHUNK_SIZE = 500_000
_NPU_CONV1D_WIDTH_LIMIT = 1_000_000


def _npu_chunked_conv1d(
    x, weight, *, stride: int, groups: int, dilation: int = 1, bias=None
):
    """Valid 1D convolution in overlapping chunks aligned to output indices.

    ``x`` has already been padded, if requested by the caller. This is
    equivalent to ``F.conv1d(x, weight, bias, stride=stride,
    dilation=dilation, groups=groups)`` apart from potential backend
    floating-point rounding differences.
    """
    # Input samples one output reads: the kernel spread out by the dilation.
    span = dilation * (weight.shape[-1] - 1) + 1
    if stride <= 0:
        raise ValueError("stride must be positive")
    if span > _NPU_CONV1D_CHUNK_SIZE:
        raise ValueError("kernel is larger than the NPU convolution chunk size")

    def conv(chunk):
        return F.conv1d(
            chunk, weight, bias, stride=stride, dilation=dilation, groups=groups
        )

    output_length = (x.shape[-1] - span) // stride + 1
    if output_length <= 0:
        # Match the usual PyTorch error for an input shorter than the kernel.
        return conv(x)

    outputs_per_chunk = (_NPU_CONV1D_CHUNK_SIZE - span) // stride + 1
    pieces = []
    for output_start in range(0, output_length, outputs_per_chunk):
        output_end = min(output_start + outputs_per_chunk, output_length)
        input_start = output_start * stride
        input_end = (output_end - 1) * stride + span
        pieces.append(conv(x[..., input_start:input_end].contiguous()))

    return torch.cat(pieces, dim=-1)


def _npu_chunked_conv_transpose1d(x, weight, *, stride: int, groups: int):
    """Transposed 1D convolution in chunks, summed over the overlaps.

    Each input sample feeds ``kernel_size`` consecutive outputs, so chunks
    taken from neighbouring input ranges overlap by ``kernel_size - stride``
    and their contributions add. Equivalent to
    ``F.conv_transpose1d(x, weight, stride=stride, groups=groups)`` apart
    from potential backend floating-point rounding differences.
    """
    kernel_size = weight.shape[-1]
    if stride <= 0:
        raise ValueError("stride must be positive")
    if kernel_size > _NPU_CONV1D_CHUNK_SIZE:
        raise ValueError("kernel is larger than the NPU convolution chunk size")

    input_length = x.shape[-1]
    output_length = (input_length - 1) * stride + kernel_size

    # Sized so that a chunk's output stays within the chunk size as well, not
    # just its input: the transposed convolution widens by roughly ``stride``.
    inputs_per_chunk = (_NPU_CONV1D_CHUNK_SIZE - kernel_size) // stride + 1

    out = None
    for input_start in range(0, input_length, inputs_per_chunk):
        input_end = min(input_start + inputs_per_chunk, input_length)
        piece = F.conv_transpose1d(
            x[..., input_start:input_end].contiguous(),
            weight,
            stride=stride,
            groups=groups,
        )
        if out is None:
            out = piece.new_zeros(*piece.shape[:-1], output_length)
        output_start = input_start * stride
        out[..., output_start : output_start + piece.shape[-1]] += piece

    return out


def _npu_too_wide(x: torch.Tensor, width: int) -> bool:
    return x.device.type == "npu" and width > _NPU_CONV1D_WIDTH_LIMIT


def npu_conv1d(module: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """``module(x)``, in chunks on NPU when the padded input is too wide.

    BigVGAN's last stage runs at the full sample rate, so its convolutions
    cross CANN's width limit at about 31 s of 32 kHz audio.
    """
    (pad,) = module.padding
    if not _npu_too_wide(x, x.shape[-1] + 2 * pad):
        return module(x)
    if module.padding_mode != "zeros":
        raise NotImplementedError(f"padding_mode {module.padding_mode!r}")
    return _npu_chunked_conv1d(
        F.pad(x, (pad, pad)),
        module.weight,
        stride=module.stride[0],
        groups=module.groups,
        dilation=module.dilation[0],
        bias=module.bias,
    )


def npu_conv_transpose1d(module: nn.ConvTranspose1d, x: torch.Tensor) -> torch.Tensor:
    """``module(x)``, in chunks on NPU when the input is too wide."""
    if not _npu_too_wide(x, x.shape[-1]):
        return module(x)
    if module.dilation[0] != 1 or module.output_padding[0] != 0:
        raise NotImplementedError("dilated or output-padded transposed convolution")
    out = _npu_chunked_conv_transpose1d(
        x, module.weight, stride=module.stride[0], groups=module.groups
    )
    (pad,) = module.padding
    if pad:
        out = out[..., pad:-pad]
    if module.bias is not None:
        out = out + module.bias.view(1, -1, 1)
    return out


if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
def kaiser_sinc_filter1d(
    cutoff, half_width, kernel_size
):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)

        weight = self.filter.expand(C, -1, -1)
        if x.device.type == "npu" and x.shape[-1] > _NPU_CONV1D_WIDTH_LIMIT:
            out = _npu_chunked_conv1d(x, weight, stride=self.stride, groups=C)
        else:
            out = F.conv1d(x, weight, stride=self.stride, groups=C)

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")

        # The limit applies to the convolution's input, not its output: this
        # layer already emits more than _NPU_CONV1D_WIDTH_LIMIT samples one
        # upsample ratio before its own input reaches it.
        weight = self.filter.expand(C, -1, -1)
        if x.device.type == "npu" and x.shape[-1] > _NPU_CONV1D_WIDTH_LIMIT:
            x = _npu_chunked_conv_transpose1d(
                x, weight, stride=self.stride, groups=C
            )
        else:
            x = F.conv_transpose1d(x, weight, stride=self.stride, groups=C)

        x.mul_(self.ratio)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x
