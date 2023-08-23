# calflops: a FLOPs and Params calculate tool for neural networks in pytorch framework
[![Pypi version](https://img.shields.io/pypi/v/calflops.svg)](https://pypi.org/project/calflops/)

This tool(calflops) is designed to compute the theoretical amount of FLOPs(floating-point operations)、MACs(multiply-add operations) and Parameters in all various neural networks, such as Linear、 CNN、 RNN、 GCN、**Transformer(Bert、LlaMA etc Large Language Model)**，including **any custom models** via ```torch.nn.function.*``` as long as based on the Pytorch implementation.

This is probably the easiest tool to calculate LLM(large language model) FLOPs, you just need ```transformers_tokenizer``` to pass in its corresponding tokenizer, and it will automatically help you build the input_shape model input. Alternatively, you can pass in the input to multiple models that you have already generated, such as input_ids, attention_mask, and so on, with the ```args```、 ```kwargs``` parameter. See the api 
of ```calflops.calculate_flops()``` for details.

In addition, the implementation process of this package inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) and [deepspeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed) libraries, Thanks for their great efforts, they are both very good work. Meanwhile this package also improves some aspects(more simple use、more model support) based on them.


This Doc is still being developed, it is pleasure for starting.

## Install the latest version
From PyPI:

```
pip install calflops
```

And you also can download latest `calflops-*-py3-none-any.whl` files from https://pypi.org/project/calflops/ 

```
pip install calflops-*-py3-none-any.whl
```

## Example
```python
from calflops import calculate_flops

# Deep Learning Model, such as alexnet.
from torchvision import models

model = models.alexnet()
batch_size = 1
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size, 3, 224, 224),
                                      output_as_string=True,
                                      output_precision=4)
print("alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#alexnet FLOPs:1.4297 GFLOPS   MACs:714.188 MMACs   Params:61.1008 M 


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
                                              transformer_tokenizer=tokenizer)
print("bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.36 GFLOPS   MACs:11.17 GMACs   Params:102.27 M 


# Large Languase Model, such as llama2-7b.
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
batch_size = 1
max_seq_length = 128
model_name = "llama2_hf_7B"
model_save = "../model/" + model_name
model = LlamaForCausalLM.from_pretrained(model_save)
tokenizer = LlamaTokenizer.from_pretrained(model_save)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=(batch_size, max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B 
```


<!-- ## Citation
if calflops was useful for your paper or tech report, please cite me:
``` python 
@online{calflops,
  author = {xiaoju ye},
  title = {calflops: a FLOPs and Params calculate tool for neural networks in pytorch framework},
  year = 2023-2023,
  url = {https://github.com/MrYxJ/calculate-flops.pytorch},
}
``` -->

## Common model calculate flops

### Large Language Model
Input data format: batch_size=1, seq_len=128

- fwd FLOPs: The FLOPs of the model forward propagation

- bwd + fwd FLOPs: The FLOPs of model forward and backward propagation

Model         | Input Shape | Params(B)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs(G) | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---         |---       |---        |--- 
bloom-1b7     |(1,128)     | 1.72B    | 1722408960  | 310.92     | 155.42   | 932.76    | 466.27
bloom-7b1     |(1,128)     | 7.07B    | 7069016064  | 1550.39    | 775.11   | 4651.18   | 2325.32
baichuan-7B   |(1,128)     | 7B       | 7000559616  | 1733.62    | 866.78   | 5200.85   | 2600.33
chatglm-6b    |(1,128)     | 6.17B    | 6173286400  | 1587.66    | 793.75   | 4762.97   | 2381.24
chatglm2-6b   |(1,128)     | 6.24B    | 6243584000  | 1537.68    | 768.8    | 4613.03   | 2306.4 
Qwen-7B       |(1,128)     | 7.72B    | 7721324544  | 1825.83    | 912.88   | 5477.48   | 2738.65
llama-7b      |(1,128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550
llama2-7b     |(1,128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550   
llama2-7b-chat |(1,128)     | 6.74B    | 6738415616  | 1700.06   | 850     | 5100.19   | 2550
chinese-llama-7b | (1,128)  | 6.89B    | 6885486592  | 1718.89    | 859.41   |5156.67   | 2578.24
chinese-llama-plus-7b| (1,128) | 6.89B | 6885486592  | 1718.89    | 859.41   |5156.67   | 2578.24
moss-moon-003-sft |(1,128) | 16.72B  | 16717980160 |  4124.93    | 2062.39  |  12374.8  | 6187.17

We can draw some simple and interesting conclusions from the table above:
- The chatglm2-6b in the model of the same scale, the model parameters are smaller, and FLOPs is also smaller, which has certain advantages in speed performance.
- The parameters of the llama1-7b, llama2-7b, and llama2-7b-chat models did not change at all, and FLOPs remained consistent. The structure of the model that conforms to the 7b described by [meta in its llama2 report](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) has not changed, the main difference is the increase of training data tokens.
- Similarly, it can be seen from the table that the chinese-llama-7b and chinese-llama-plus-7b data are also in line with [cui's report](https://arxiv.org/pdf/2304.08177v1.pdf), just more chinese data tokens are added for training, and the model structure and parameters do not change.

- ......

More model FLOPs would be updated successively, see github [calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch)

### Bert

Input data format: batch_size=1, seq_len=128

Model         | Input Shape | Params(M)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs() | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---        |---       |---     |---
hfl/chinese-roberta-wwm-ext | (1,128)| 102.27M | 102267648 |       67.1  |    33.52  |  201.3 | 100.57
......

You can use calflops to calculate the more different model based bert, look forward to updating in this form.


## Benchmark
### [torchvision](https://pytorch.org/docs/1.0.0/torchvision/models.html)

Model         | Input Resolution | Params(M)|Params(Total) | FLOPs(G) | FLOPs(Total) | Macs(G) | Macs(Total) 
---           |---               |---        |---          |---     |---          |---     |---
alexnet       |224x224           | 61.10     | 61100840    | 1.43   | 1429740000  | 741.19 | 7418800000
vgg11         |224x224           | 132.86    | 132863000   | 15.24  | 15239200000 | 7.61   | 7609090000
vgg13         |224x224           | 133.05    | 133048000   | 22.65  | 22647600000 | 11.31  | 11308500000
vgg16         |224x224           | 138.36    | 138358000   | 30.97  | 30973800000 | 15.47  | 15470300000
vgg19         |224x224           | 143.67    | 143667000   | 39.30  | 39300000000 | 19.63  | 19632100000
vgg11_bn      |224x224           | 132.87    | 132869000   | 15.25  | 15254000000 | 7.61   | 7609090000
vgg13_bn      |224x224           | 133.05    | 133054000   | 22.67  | 22672100000 | 11.31  | 11308500000
vgg16_bn      |224x224           | 138.37    | 138366000   | 31.00  | 31000900000 | 15.47  | 15470300000
vgg19_bn      |224x224           | 143.68    | 143678000   | 39.33  | 39329700000 | 19.63  | 19632100000
resnet18      |224x224           | 11.69     | 11689500    | 3.64   | 3636250000  | 1.81   | 1814070000
resnet34      |224x224           | 21.80     | 21797700    | 7.34   | 7339390000  | 3.66   | 3663760000
resnet50      |224x224           | 25.56     | 25557000    | 8.21   | 8211110000  | 4.09   | 4089180000
resnet101     |224x224           | 44.55     | 44549200    | 15.65  | 15690900000 | 7.80   | 7801410000
resnet152     |224x224           | 60.19     | 60192800    | 23.09  | 23094300000 | 11.51  | 11513600000
squeezenet1_0 |224x224           | 1.25      | 1248420     | 1.65   | 1648970000  | 0.82   | 818925000
squeezenet1_1 |224x224           | 1.24      | 1235500     | 0.71   | 705014000   | 0.35   | 349152000
densenet121   |224x224           | 7.98      | 7978860     | 5.72   | 5716880000  | 2.83   | 2834160000
densenet169   |224x224           | 14.15     | 14195000    | 6.78   | 6778370000  | 3.36   | 3359840000
densenet201   |224x224           | 20.01     | 20013900    | 8.66   | 8658520000  | 4.29   | 4291370000
densenet161   |224x224           | 28.68     | 28681000    | 15.55  | 1554650000  | 7.73   | 7727900000
inception_v3  |224x224           | 27.16     | 27161300    | 5.29   | 5692390000  | 2.84   | 2837920000
 
You also can compare torchvision results of calculate FLOPs with anthoer good tool: [ptflops readme.md](https://github.com/sovrasov/flops-counter.pytorch/).

<!-- ### [torchvision](https://pytorch.org/docs/1.0.0/torchvision/models.html)

Input data format: batch_size = 1

actually input_shape = (1, 3, 224, 224)

Note: The FLOPs in the table only takes into account the computation of forward propagation of the model, **Total** refers to the total numerical representation without unit abbreviations.


Model         | Input Resolution | Params(M)|Params(Total) | FLOPs(G) | FLOPs(Total) | Macs(G) | Macs(Total) 
---           |---               |---        |---      |---          |---      |---    |---
alexnet       |224x224           | 61.1      | 61100840    | 43.45   | 20.91
vgg11         |224x224           | 132.86    | 7.63    | 30.98       | 11.37
vgg13         |224x224           | 133.05    | 11.34   | 30.07       | 10.75
vgg16         |224x224           | 138.36    | 15.5    | 28.41       | 9.62
vgg19         |224x224           | 143.67    | 19.67   | 27.62       | 9.12
vgg11_bn      |224x224           | 132.87    | 7.64    | 29.62       | 10.19
vgg13_bn      |224x224           | 133.05    | 11.36   | 28.45       | 9.63
vgg16_bn      |224x224           | 138.37    | 15.53   | 26.63       | 8.50
vgg19_bn      |224x224           | 143.68    | 19.7    | 25.76       | 8.15
resnet18      |224x224           | 11.69     | 1.82    | 30.24       | 10.92
resnet34      |224x224           | 21.8      | 3.68    | 26.70       | 8.58
resnet50      |224x224           | 25.56     | 4.12    | 23.85       | 7.13
resnet101     |224x224           | 44.55     | 7.85    | 22.63       | 6.44
resnet152     |224x224           | 60.19     | 11.58   | 21.69       | 5.94
squeezenet1_0 |224x224           | 1.25      | 0.83    | 41.90       | 19.58
squeezenet1_1 |224x224           | 1.24      | 0.36    | 41.81       | 19.38
densenet121   |224x224           | 7.98      | 2.88    | 25.35       | 7.83
densenet169   |224x224           | 14.15     | 3.42    | 24.00       | 7.00
densenet201   |224x224           | 20.01     | 4.37    | 22.80       | 6.43
densenet161   |224x224           | 28.68     | 7.82    | 22.35       | 6.20
inception_v3  |224x224           | 27.16     | 2.85    | 22.55       | 6.44 -->



## calculate_flops API

```python
def calculate_flops(model,
                    input_shape=None,
                    transformer_tokenizer=None,
                    args=[],   
                    kwargs={},
                    forward_mode="forward",
                    include_backPropagation=False,
                    compute_bp_factor=2.0,         
                    print_results=True,
                    print_detailed=True,
                    output_as_string=True,
                    output_precision=2,
                    output_unit=None,
                    ignore_modules=None):
    
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model ([torch.nn.Module]): The model of input must be a PyTorch model.
        input_shape (tuple, optional): Input shape to the model. If args and kwargs is empty, the model takes a tensor with this shape as the only positional argument. Default to [].
        transformers_tokenizer (None, optional): Transforemrs Toekenizer must be special if model type is transformers and args、kwargs is empty. Default to None
        args (list, optinal): list of positional arguments to the model, such as bert input args is [input_ids, token_type_ids, attention_mask]. Default to []
        kwargs (dict, optional): dictionary of keyword arguments to the model, such as bert input kwargs is {'input_ids': ..., 'token_type_ids':..., 'attention_mask':...}. Default to {}
        forward_mode (str, optional): To determine the mode of model inference, Default to 'forward'. And use 'generate' if model inference uses model.generate().
        include_backPropagation (bool, optional): Decides whether the final return FLOPs computation includes the computation for backpropagation.
        compute_bp_factor (float, optional): The model backpropagation is a multiple of the forward propagation computation. Default to 2.
        print_results (bool, optional): Whether to print the model profile. Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. Defaults to True.
        output_precision (int, optional) : Output holds the number of decimal places if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value, such as T, G, M, and K. Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
```


## Concact Author

Author: [MrYXJ](https://github.com/MrYxJ/)

Mail: yxj2017@gmail.com
