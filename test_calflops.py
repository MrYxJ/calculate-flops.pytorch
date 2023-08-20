# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : code_job@163.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 13:05:48
 LastEditTime : 2023-08-20 23:59:07
 Copyright (C) 2023 mryxj. All rights reserved.
'''
 
from torchvision import models

from calflops.flops_counter import calculate_flops_pytorch

model = models.alexnet()
batch_size = 1
flops, macs, params = calculate_flops_pytorch(model=model, \
                        input_shape=(batch_size, 3, 224, 224))
print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

