# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 10:27:55
 LastEditTime : 2023-09-05 15:31:43
 Copyright (C) 2023 mryxj. All rights reserved.
'''

from .flops_counter import calculate_flops
from .flops_counter_hf import calculate_flops_hf

from .utils import generate_transformer_input
from .utils import number_to_string
from .utils import flops_to_string
from .utils import macs_to_string
from .utils import params_to_string
from .utils import bytes_to_string

from .estimate import create_empty_model