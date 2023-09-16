# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 11:01:23
 LastEditTime : 2023-09-05 15:51:50
 Copyright (C) 2023 mryxj. All rights reserved.
'''

import importlib

import torch

DEFAULT_PRECISION = 2


# def generate_transformer_input(model_tokenizer, input_shape, device):
#     """Automatically generates data in the form of transformes model input format.

#     Args:
#         input_shape (tuple):transformers model input shape: (batch_size, seq_len).
#         tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

#     Returns:
#         dict: data format of transformers model input, it is a dict contain 'input_ids', 'attention_mask', sometime contain 'token_type_ids'.
#     """

#     if input_shape is None:
#         input_shape = [1, 128] # defautl (batch_size=1, seq_len=128)

#     max_length = input_shape[1]
#     model_input_ids = []
#     model_attention_mask = []
#     model_token_type_ids = []
#     model_position_ids = []

#     import numpy as np
#     inp_seq = ""
#     for _ in range(input_shape[0]):
#         inputs = model_tokenizer.encode_plus(
#             inp_seq,
#             add_special_tokens=True,
#             truncation_strategy='longest_first',
#         )
#         origin_length = len(inputs["input_ids"])
#         padding_length = max_length - origin_length

#         for key in inputs.keys():
#             if key == "input_ids":
#                 input_ids = inputs["input_ids"]
#                 pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
#                 input_ids = input_ids + ([pad_token] * padding_length)
#                 assert len(input_ids) == max_length,  "len(input_ids) must equal max_length"
#                 model_input_ids.append(input_ids)
#             elif key == "attention_mask":
#                 attention_mask = [1] * origin_length
#                 attention_mask = attention_mask + ([0] * padding_length)
#                 assert len(attention_mask) == max_length, "len(attention_mask) must equal max_length"
#                 model_attention_mask.append(attention_mask)
#             elif key == "token_type_ids":
#                 token_type_ids = inputs['token_type_ids']
#                 pad_token_segment_id = 0
#                 token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
#                 assert len(token_type_ids) == max_length,  "len(token_type_ids) must equal max_length"
#                 model_token_type_ids.append(token_type_ids)
#             elif key == "position_ids":                      
#                 position_ids = inputs['position_ids']
#                 if isinstance(position_ids, list):
#                     for i in range(origin_length, max_length):
#                         position_ids.append(i)
#                     assert len(position_ids) == max_length,  "len(position_ids) must equal max_length"
#                 elif isinstance(position_ids, np.ndarray):
#                     pass
#                 model_position_ids.append(position_ids)

#     # Batch size input_shape[0], sequence length input_shape[128]
#     inputs = {}
#     if len(model_input_ids) > 0:
#         inputs.update({"input_ids": torch.tensor(model_input_ids).to(device)})
#     if len(model_attention_mask) > 0 and not isinstance(model_attention_mask[0], list):
#         inputs.update({"attention_mask": torch.tensor(model_attention_mask).to(device)})
#     if len(model_token_type_ids) > 0:  
#         inputs.update({'token_type_ids': torch.tensor(model_token_type_ids).to(device)})
#     if len(model_position_ids) > 0 and not isinstance(model_position_ids[0], np.ndarray):
#         inputs.update({'position_ids': torch.tensor(model_position_ids).to(device)})

#     return inputs

def generate_transformer_input(model_tokenizer, input_shape, device):
    """Automatically generates data in the form of transformes model input format.
    
    Args:
        input_shape (tuple):transformers model input shape: (batch_size, seq_len).
        tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

    Returns:
        dict: data format of transformers model input, it is a dict contain 'input_ids', 'attention_mask', sometime contain 'token_type_ids'.
    """

    if input_shape is None:
        input_shape = [1, 128]  # defautl (batch_size=1, seq_len=128)

    max_length = input_shape[1]
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    model_position_ids = []

    inp_seq = ""
    for _ in range(input_shape[0]):
        inputs = model_tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        origin_length = len(inputs["input_ids"])
        padding_length = max_length - origin_length

        for key in inputs.keys():
            if key == "input_ids":
                input_ids = inputs["input_ids"]
                pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
                input_ids = input_ids + ([pad_token] * padding_length)
                assert len(input_ids) == max_length, "len(input_ids) must equal max_length"
                model_input_ids.append(input_ids)
            elif key == "attention_mask":
                attention_mask = [1] * origin_length
                attention_mask = attention_mask + ([0] * padding_length)
                assert len(attention_mask) == max_length, "len(attention_mask) must equal max_length"
                model_attention_mask.append(attention_mask)
            elif key == "token_type_ids":
                token_type_ids = inputs['token_type_ids']
                pad_token_segment_id = 0
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(token_type_ids) == max_length, "len(token_type_ids) must equal max_length"
                model_token_type_ids.append(token_type_ids)
            elif key == "position_ids":  # chatglm2 use position id
                position_ids = inputs['position_ids']
                for i in range(origin_length, max_length):
                    position_ids.append(i)
                assert len(position_ids) == max_length, "len(position_ids) must equal max_length"
                model_position_ids.append(position_ids)

    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {}
    if len(model_input_ids) > 0:
        inputs.update({"input_ids": torch.tensor(model_input_ids).to(device)})
    if len(model_attention_mask) > 0:
        inputs.update({"attention_mask": torch.tensor(model_attention_mask).to(device)})
    if len(model_token_type_ids) > 0:
        inputs.update({'token_type_ids': torch.tensor(model_token_type_ids).to(device)})
    if len(model_position_ids) > 0:
        inputs.update({'position_ids': torch.tensor(model_position_ids).to(device)})

    return inputs


def number_to_string(num, units=None, precision=DEFAULT_PRECISION):
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1
    return f"{round(num / magnitude, precision):g} {units}"


def macs_to_string(macs, units=None, precision=DEFAULT_PRECISION):
    """Converts macs in numeric form to string form.

    Args:
        macs (int): Calculate the results of the model macs in numerical form.
        units (str, optional): The unit of macs after conversion to string representation, such as TMACs、GMACs、MMACs、KMACs
        precision (int, optional): The number of digits of the result is preserved. Defaults to DEFAULT_PRECISION.

    Returns:
        string: The string representation of macs.
    """
    return f"{number_to_string(macs, units=units, precision=precision)}MACs"


def flops_to_string(flops, units=None, precision=DEFAULT_PRECISION):
    """Converts flops in numeric form to string form.

    Args:
        flops (int): Calculate the results of the model flops in numerical form.
        units (str, optional): The unit of flops after conversion to string representation, such as TFLOPs,GFLOPs,MFLOPs,KFLOPs.
        precision (int, optional): The number of digits of the result is preserved. Defaults to DEFAULT_PRECISION.

    Returns:
        string: The string representation of flops.
    """
    return f"{number_to_string(flops, units=units, precision=precision)}FLOPS"


def bytes_to_string(b, units=None, precision=DEFAULT_PRECISION):
    """Converts bytes in numeric form to string form.

    Args:
        b (int): Calculate the results of the bytes in numerical form.
        units (str, optional): The unit of bytes after conversion to string representation, such as TB,GB,MB,KB.
        precision (int, optional): The number of digits of the result is preserved. Defaults to DEFAULT_PRECISION.

    Returns:
        string: The string representation of bytes.
    """
    return f"{number_to_string(b, units=units, precision=precision)}B"


def params_to_string(params_num, units=None, precision=DEFAULT_PRECISION):
    """Converts params in numeric form to string form.

    Args:
        params_num (int): Calculate the results of the model param in numerical form.
        units (str, optional): The unit of params after conversion to string representation.
        precision (int, optional): The number of digits of the result is preserved. Defaults to DEFAULT_PRECISION.

    Returns:
        string: The string representation of params.
    """
    units = units.replace("B", "G") if units else units
    return number_to_string(params_num, units=units, precision=precision).replace("G", "B").strip()


def get_module_flops(module, is_sparse=False):
    """Recursively compute the FLOP s of the model

    Args:
        module (pytorch module): model format must be pytorch
        is_sparse (bool, Optional): Whether to exclude sparse weight. Defaults to False.

    Returns:
        int: The sum of the entire model flops
    """
    sum_flops = module.__flops__ * sum(
        p.count_nonzero().item() for p in module.parameters() if p.requires_grad
    ) / sum(p.numel() for p in module.parameters() if p.requires_grad) if is_sparse else module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum_flops += get_module_flops(child, is_sparse=is_sparse)
    return sum_flops


def get_module_macs(module, is_sparse=False):
    """Recursively compute the macs s of the model

    Args:
        module (pytorch module): model format must be pytorch
        is_sparse (bool, Optional): Whether to exclude sparse weight. Defaults to False.

    Returns:
        int: The sum of the entire model macs
    """
    sum_macs = module.__macs__ * sum(
        p.count_nonzero().item() for p in module.parameters() if p.requires_grad
    ) / sum(p.numel() for p in module.parameters() if p.requires_grad) if is_sparse else module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum_macs += get_module_macs(child, is_sparse=is_sparse)
    return sum_macs


def convert_bytes(size):
    "Converts `size` from bytes to the largest possible unit"
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{round(size, 2)} {x}"
        size /= 1024.0

    return f"{round(size, 2)} PB"


def _is_package_available(pkg_name):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
