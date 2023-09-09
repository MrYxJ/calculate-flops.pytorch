# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 13:05:48
 LastEditTime : 2023-09-09 00:26:18
 Copyright (C) 2023 mryxj. All rights reserved.
'''
# import os 
# os.system("pip install calflops")

from calflops import calculate_flops
from torchvision import models

model = models.alexnet()
batch_size = 1

# output
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size, 3, 224, 224),
                                      output_as_string=False,
                                      print_results=True,
                                      print_detailed=True)
print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

# 
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size, 3, 224, 224),
                                      print_results=False,
                                      print_detailed=False,
                                      output_as_string=True,
                                      output_precision=3,
                                      output_unit='M')
print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))