# calculate-flops.pytorch
This tool(calflops) is designed to compute the theoretical amount of FLOPs(floating-point operations)、MACs(multiply-add operations) and Parameters in all various neural networks, such as Linear、 CNN、 RNN、 GCN、**Transformer(Bert、LlaMA etc Large Language Model)**，including **any custom models** via ```torch.nn.function.*``` as long as based on the Pytorch implementation.

In addition, the implementation process of this package inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) and [deepspeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed) libraries, for which I am very grateful for their great efforts, they are both very good work. Meanwhile this package also improves some aspects(more simple use、more model support) based on them.


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

### large language model

Input data format: batch_size=1, seq_len=128

fwd FLOPs: The FLOPs of the model forward propagation

bwd + fwd FLOPs: The FLOPs of model forward and backward propagation

Model         | Input Shape | Params(B)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs(G) | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---         |---       |---        |--- 
baichuan-7B   |(1, 128)     | 7B       | 7000559616  | 1733.62    | 866.78   | 5200.85   | 2600.33
chatglm-6b    |(1, 128)     | 6.17B    | 6173286400  | 1587.66    | 793.75   | 4762.97   | 2381.24
chatglm2-6b   |(1, 128)     | 6.24B    | 6243584000  | 1537.68    | 768.8    | 4613.03   | 2306.4 
falcon-7b     |(1, 128)  | | | | |
falcon-7b-instruct |(1,128) | | | |
Qwen-7B       |(1, 128)     | 7.72B    | 7721324544  | 1825.83    | 912.88   | 5477.48   | 2738.65
Qwen-7B-Chat  |(1, 128)     |          |             |            |          |           | 
llama-7b      |(1, 128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550
llama2-7b     |(1, 128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550   
llama2-7b-chat |(1, 128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550   
moss-moon-003-base |(1, 128) |   |             |            |          |           | 
moss-moon-003-sft |(1, 128) | 16.72B  | 16717980160 |  4124.93 |    2062.39  |  12374.8 | 6187.17


### transformers

Input data format: batch_size=1, seq_len=128

Model         | Input Shape | Params(M)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs() | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---        |---       |---     |---
hfl/chinese-roberta-wwm-ext | (1,128)| 102.27M | 102267648 |       67.1  |    33.52  |  201.3 | 100.57
......




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

Mail: code.mryxj@gmail.com