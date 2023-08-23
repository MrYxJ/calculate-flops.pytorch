# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : code_job@163.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 13:05:48
 LastEditTime : 2023-08-22 16:01:54
 Copyright (C) 2023 mryxj. All rights reserved.
'''
 
from torchvision import models

from calflops.flops_counter import calculate_flops

model = models.alexnet()
batch_size = 1
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size, 3, 224, 224),
                                      output_as_string=True,
                                      output_precision=2,
                                      output_unit='M',
                                      print_results=True,
                                      print_detailed=True)
print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

