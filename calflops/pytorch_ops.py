# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 22:34:47
 LastEditTime : 2023-08-23 11:17:33
 Copyright (C) 2023 mryxj. All rights reserved.
'''

'''
The part of code is inspired by ptflops and deepspeed profiling.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from typing import Optional
from collections import OrderedDict

Tensor = torch.Tensor


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p

def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs

# Activation just calculate FLOPs， MACs is 0
def _relu_flops_compute(input, inplace=False):
    return input.numel(), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel(), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return input.numel(), 0


def _leaky_relu_flops_compute(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    return input.numel(), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _gelu_flops_compute(input, **kwargs):
    return input.numel(), 0


def _pool_flops_compute(input,
                        kernel_size,
                        stride=None,
                        padding=0,
                        dilation=None,
                        ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=None,
                        return_indices=None):
    return input.numel(), 0


def _conv_flops_compute(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length
    if isinstance(padding, str):
        if padding == 'valid':
            paddings = (0, ) * length
        elif padding == 'same':
            paddings = ()
            for d, k in zip(dilations, kernel_dims):
                total_padding = d * (k - 1)
                paddings += (total_padding // 2, )
    elif isinstance(padding, tuple):
        paddings = padding
    else:
        paddings = (padding, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
                                                       (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[1]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)
     
    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
                                                       (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: Optional[Tensor] = None,
                              bias: Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _upsample_flops_compute(*args, **kwargs):
    input = args[0]
    size = kwargs.get('size', None)
    if size is None and len(args) > 1:
        size = args[1]

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0

    scale_factor = kwargs.get('scale_factor', None)
    if scale_factor is None and len(args) > 2:
        scale_factor = args[2]
    assert scale_factor is not None, "either size or scale_factor should be defined"

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel(), 0


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def wrapFunc(func, funcFlopCompute, old_functions, module_flop_count, module_mac_count):
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return oldFunc(*args, **kwds)

    newFunc.__str__ = func.__str__

    return newFunc


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    gates_size = w_ih.shape[0]
    # matrix matrix mult ih state and internal state
    flops += 2 * w_ih.shape[0] * w_ih.shape[1] - gates_size
    # matrix matrix mult hh state and internal state
    flops += 2 * w_hh.shape[0] * w_hh.shape[1] - gates_size
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}

def _patch_functionals(old_functions, module_flop_count, module_mac_count):
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute, old_functions, module_flop_count, module_mac_count)
    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute, old_functions, module_flop_count, module_mac_count)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute, old_functions, module_flop_count, module_mac_count)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.elu = wrapFunc(F.elu, _elu_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute, old_functions, module_flop_count, module_mac_count)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute, old_functions, module_flop_count, module_mac_count)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute, old_functions, module_flop_count, module_mac_count)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute, old_functions, module_flop_count, module_mac_count)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute, old_functions, module_flop_count, module_mac_count)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute, old_functions, module_flop_count, module_mac_count)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute, old_functions, module_flop_count, module_mac_count)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute, old_functions, module_flop_count, module_mac_count)


def _patch_tensor_methods(old_functions, module_flop_count, module_mac_count):
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)
    # torch.mm = wrapFunc(torch.mm, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)
    # torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)
    # torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)
    # torch.Tensor.bmm = wrapFunc(torch.Tensor.bmm, _matmul_flops_compute, old_functions, module_flop_count, module_mac_count)

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute, old_functions, module_flop_count, module_mac_count)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute, old_functions, module_flop_count, module_mac_count)

    torch.mul = wrapFunc(torch.mul, _mul_flops_compute, old_functions, module_flop_count, module_mac_count)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute, old_functions, module_flop_count, module_mac_count)

    torch.add = wrapFunc(torch.add, _add_flops_compute, old_functions, module_flop_count, module_mac_count)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute, old_functions, module_flop_count, module_mac_count)

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute, old_functions, module_flop_count, module_mac_count)

    torch.baddbmm = wrapFunc(torch.baddbmm, _tensor_addmm_flops_compute, old_functions, module_flop_count, module_mac_count)


def _reload_functionals(old_functions):
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.embedding = old_functions[F.embedding.__str__]


def _reload_tensor_methods(old_functions):
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    # torch.mm = old_functions[torch.mm.__str__]
    # torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    # torch.bmm = old_functions[torch.matmul.__str__]
    # torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]
    torch.einsum = old_functions[torch.einsum.__str__]
    torch.baddbmm = old_functions[torch.baddbmm.__str__]
