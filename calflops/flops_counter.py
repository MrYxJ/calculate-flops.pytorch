# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 10:28:55
 LastEditTime : 2023-08-21 00:33:00
 Copyright (C) 2023 mryxj. All rights reserved.
'''


import torch
import torch.nn as nn

from .utils import generate_transformer_input
from .utils import flops_to_string
from .utils import macs_to_string
from .utils import params_to_string

from .calculate_pipline import CalFlopsPipline

def calculate_flops_pytorch(model,
                    input_shape=None,
                    transformer_tokenizer=None,
                    args=[],   # [input_ids, token_type_ids, attention_mask]
                    kwargs={}, # {'input_ids': ..., 'token_type_ids':..., 'attention_mask':...}
                    forward_mode="forward", # foward_mode 区分 dl 与 llm 中不同的推理方式。
                    include_backPropagation=False, # 最后返回FLOPs计算量是否包括反向传播
                    compute_bp_factor=2.0,         # FLOPs计算量默认反向传播是正向的两倍
                    print_results=True,
                    print_detailed=True,
                    as_string=True,
                    ignore_modules=None):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:
    .. code-block:: python
        # Deep Learning Model, such as alexnet.
        from torchvision import models

        model = models.alexnet()
        batch_size = 1
        flops, macs, params = calculate_flops_pytorch(model=model, 
                                                      input_shape=(batch_size, 3, 224, 224),
                                                      print_results=False)
        print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        # alexnet FLOPs:1.43 GFLOPS   MACs:714.188 MMACs   Params:61.101 M 
        
        # Transformers Model, such as bert.
        from transformers import AutoModel
        from transformers import AutoTokenizer
        batch_size = 1
        max_seq_length = 128
        model_name = "hfl/chinese-roberta-wwm-ext/"
        model_save = "../pretrain_models/" + model_name
        model = AutoModel.from_pretrained(model_save)
        tokenizer = AutoTokenizer.from_pretrained(model_save)
        flops, macs, params = calculate_flops_pytorch(model=model, 
                                                      input_shape=(batch_size, max_seq_length),
                                                      transformer_tokenizer=tokenizer,
                                                      print_results=False)
        print("bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        # bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.363 GFLOPS   MACs:11.174 GMACs   Params:102.268 M 
        
        # Large Languase Model, such as llama-7b.
        from transformers import LlamaTokenizer
        from transformers import LlamaForCausalLM
        batch_size = 1
        max_seq_length = 128
        model_name = "original_llama2_hf_7B"
        model_save = "../model/" + model_name
        model = LlamaForCausalLM.from_pretrained(model_save)
        tokenizer = LlamaTokenizer.from_pretrained(model_save)
        flops, macs, params = calculate_flops_pytorch(model=model,
                                                      input_shape=(batch_size, max_seq_length),
                                                      transformer_tokenizer=tokenizer,
                                                      print_results=False)
        print("llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        # llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.001 GMACs   Params:6.738 B 

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        transformers_tokenizer (None): 
        auto_generate_transformer_input (bool): 
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        forward_mode (str): 
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    #assert transformers_tokenizer and auto_generate_transformers_input and "transformers" in str(type(model)), "The model must be a transformers model if args of auto_generate_transformers_input is True and transformers_tokenizer is not None"
    model.eval()

    is_Transformer = True if "transformers" in str(type(model)) else False

    if input_shape is not None:
        assert len(args) == 0 and len(kwargs) == 0, "args and kwargs must be empty value if input_shape is not None, then will be generate random input by inpust_shape"
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"

        calculate_flops_pipline = CalFlopsPipline(model=model, 
                                              include_backPropagation=include_backPropagation,
                                              compute_bp_factor=compute_bp_factor)
        calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)
        
        if transformer_tokenizer is None:  # model is not transformers model
            assert is_Transformer is False, "the model is must not transformer model if input_shape is not None and transformer_tokenizer is None"
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape, ),
                    dtype=next(model.parameters()).dtype,
                    device=next(model.parameters()).device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape, ))

            args = [input]
        else:
            assert len(input_shape) == 2, "the format of input_shape must be (batch_size, seq_len) if model is transformers model and auto_generate_transformers_input if True"
            kwargs = generate_transformer_input(input_shape=input_shape, model_tokenizer=transformer_tokenizer)
    else:
        assert transformer_tokenizer or (len(args) > 0 or len(kwargs) > 0),  "input_shape or args or kwargs one of there parameters must specified if auto_generate_input is False"
        if transformer_tokenizer:
            kwargs = generate_transformer_input(input_shape=None, model_tokenizer=transformer_tokenizer)
    
    if kwargs:
        if forward_mode == 'forward':
            _ = model(*args, **kwargs)
        if forward_mode == 'generate':
            _ = model.generate(*args, **kwargs)
    else:
        if forward_mode == 'forward':
            _ = model(*args)
        if forward_mode == 'generate':
            _ = model.generate(*args)

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()
    
    if print_results:
        calculate_flops_pipline.print_model_pipline(detailed=print_detailed)

    calculate_flops_pipline.end_flops_calculate()
   
    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor) 
        macs = macs * (1 + compute_bp_factor)

    if as_string:
        return flops_to_string(flops), macs_to_string(macs), params_to_string(params)
    
    return flops, macs, params