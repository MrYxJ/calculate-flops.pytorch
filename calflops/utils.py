# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : code_job@163.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 11:01:23
 LastEditTime : 2023-08-21 00:25:39
 Copyright (C) 2023 mryxj. All rights reserved.
'''

import torch

DEFAULT_PRECISION = 3


def generate_transformer_input(model_tokenizer, input_shape=None):
    """Automatically generates data in the form of transformes model input format.
    
    Args:
        input_shape (tuple):transformers model input shape: (batch_size, seq_len).
        tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

    Returns:
        dict: data format of transformers model input, it is a dict contain 'input_ids', 'attention_mask', sometime contain 'token_type_ids'.
    """
    if input_shape is None:
        input_shape = [1, 128] # defautl (batch_size=1, seq_len=128)
    
    max_length = input_shape[1]
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []

    inp_seq = ""   
    
    for _ in range(input_shape[0]):
        inputs = model_tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
        pad_token_segment_id = 0
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        model_input_ids.append(input_ids)
        model_attention_mask.append(attention_mask)
        
        if "token_type_ids" in inputs.keys():         # LLM tokenizer don't have token_type_ids
            token_type_ids = inputs['token_type_ids']
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            assert len(token_type_ids) == max_length
            model_token_type_ids.append(token_type_ids)
   
    # labels = torch.tensor([1] * input_shape[0])
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {
        "input_ids": torch.tensor(model_input_ids),
        "attention_mask": torch.tensor(model_attention_mask),
    }
    if len(model_token_type_ids) > 0:        # LLM tokenizer don't have token_type_ids 
        inputs.update({'token_type_ids': torch.tensor(model_token_type_ids)})
    #inputs.update({"labels": labels})
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
    return f"{number_to_string(macs, units=units, precision=precision)}MACs"


def flops_to_string(flops, units=None, precision=DEFAULT_PRECISION):
    return f"{number_to_string(flops, units=units, precision=precision)}FLOPS"


def bytes_to_string(b, units=None, precision=DEFAULT_PRECISION):
    return f"{number_to_string(b, units=units, precision=precision)}B"


def params_to_string(params_num, units=None, precision=DEFAULT_PRECISION):
    units = units.replace("B", "G") if units else units
    return number_to_string(params_num, units=units, precision=precision).replace("G", "B").strip()

def get_module_flops(module):
    """Recursively compute the FLOP s of the model
    """
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_flops(child)
    return sum

def get_module_macs(module):
    """Recursively compute the macs s of the model
    """
    sum = module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_macs(child)
    return sum
