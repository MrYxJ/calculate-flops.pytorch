# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-09-03 11:03:58
 LastEditTime : 2023-09-09 15:17:53
 Copyright (C) 2023 mryxj. All rights reserved.
'''


import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .utils import generate_transformer_input
from .utils import flops_to_string
from .utils import macs_to_string
from .utils import params_to_string
from .estimate import create_empty_model
from .calculate_pipline import CalFlopsPipline


def calculate_flops_hf(model_name,
                       empty_model=None,
                       input_shape=None,
                       trust_remote_code=True,
                       access_token="",
                       forward_mode="forward",
                       include_backPropagation=False,
                       compute_bp_factor=2.0,
                       print_results=True,
                       print_detailed=True,
                       output_as_string=True,
                       output_precision=2,
                       output_unit=None,
                       ignore_modules=None,
                       return_results=False):
    
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model_name (str): The model name in huggingface platform https://huggingface.co/models, such as meta-llama/Llama-2-7b、baichuan-inc/Baichuan-13B-Chat etc.
        input_shape (tuple, optional): Input shape to the model. If args and kwargs is empty, the model takes a tensor with this shape as the only positional argument. Default to [].
        trust_remote_code (bool, optional): Trust the code in the remote library for the model structure.
        access_token (str, optional): Some models need to apply for a access token, such as meta llama2 etc.
        forward_mode (str, optional): To determine the mode of model inference, Default to 'forward'. And use 'generate' if model inference uses model.generate().
        include_backPropagation (bool, optional): Decides whether the final return FLOPs computation includes the computation for backpropagation.
        compute_bp_factor (float, optional): The model backpropagation is a multiple of the forward propagation computation. Default to 2.
        print_results (bool, optional): Whether to print the model profile. Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. Defaults to True.
        output_precision (int, optional) : Output holds the number of decimal places if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value, such as T, G, M, and K. Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
        
    Example:
    .. code-block:: python
    from calflops import calculate_flops_hf
    
    batch_size = 1
    max_seq_length = 128
    model_name = "baichuan-inc/Baichuan-13B-Chat"
    flops, macs, params = calculate_flops_hf(model_name=model_name,
                                            input_shape=(batch_size, max_seq_length))
    print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    
    if empty_model == None:
        empty_model = create_empty_model(model_name=model_name,
                                         library_name=None,
                                         trust_remote_code=trust_remote_code,
                                         access_token=access_token)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code,
                                              access_token=access_token)
    
    assert isinstance(empty_model, nn.Module), "model must be a PyTorch module"
    device = next(empty_model.parameters()).device
    empty_model = empty_model.to(device)
    empty_model.eval()
    
    calculate_flops_pipline = CalFlopsPipline(model=empty_model,
                                              include_backPropagation=include_backPropagation,
                                              compute_bp_factor=compute_bp_factor)
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        assert len(input_shape) == 2, "the format of input_shape must be (batch_size, seq_len) if model is transformers model and auto_generate_transformers_input if True"
        kwargs = generate_transformer_input(input_shape=input_shape,
                                            model_tokenizer=tokenizer,
                                            device=device)
    else:
        kwargs = generate_transformer_input(input_shape=None,
                                            model_tokenizer=tokenizer,
                                            device=device)
    
    for key, value in kwargs.items():
        kwargs[key] = value.to(device)
    
    try:
        if forward_mode == 'forward':
            _ = empty_model(**kwargs)
        if forward_mode == 'generate':
            _ = empty_model.generate(**kwargs)
    except Exception as e:
        ErrorInformation = """The model:%s meet a problem in forwarding, perhaps because the model:%s cannot be deduced on meta device. 
        You can downloaded complete model parameters in locally from huggingface platform, and then using another function:calflops.calculate_flops(model, tokenizer) to calculate FLOPs on the gpu device.\n
        Error Information: %s\n.
        """ % (model_name, model_name, e)
        print(ErrorInformation)
        return None, None, None
    else:
        flops = calculate_flops_pipline.get_total_flops()
        macs = calculate_flops_pipline.get_total_macs()
        params = calculate_flops_pipline.get_total_params()

  
        print_return = calculate_flops_pipline.print_return_model_pipline(units=output_unit,
                                                    precision=output_precision,
                                                    print_detailed=print_detailed,
                                                    print_results=print_results)
        
        calculate_flops_pipline.end_flops_calculate()
        
        if include_backPropagation:
            flops = flops * (1 + compute_bp_factor)
            macs = macs * (1 + compute_bp_factor)
        
        if output_as_string:
            flops = flops_to_string(flops, units=output_unit, precision=output_precision)
            macs = macs_to_string(macs, units=output_unit, precision=output_precision)
            params = params_to_string(params, units=output_unit, precision=output_precision)
        
        if return_results:
            return flops, macs, params, print_return
        else:
            return flops, macs, params


