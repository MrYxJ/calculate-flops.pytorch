<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  calflops: a FLOPs and Params calculate tool for neural networks
</h1>
</div>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/calflops)
[![Pypi version](https://img.shields.io/pypi/v/calflops.svg)](https://pypi.org/project/calflops/)
[![PyPI - License](https://img.shields.io/pypi/l/calflops)](https://github.com/MrYxJ/calculate-flops.pytorch/blob/main/LICENSE)

<h4 align="center">
    <p>
        <a href="https://github.com/MrYxJ/calculate-flops.pytorch">English</a>|
        <b>中文</b> 
    <p>
</h4>

# 介绍
这个工具(calflops)的作用是通过对模型结构与实现上统计计算各种神经网络中的FLOPs(浮点运算)，mac(乘加运算)和模型参数的理论量，支持模型包括：Linear, CNN, RNN, GCN， **Transformer(Bert, LlaMA等大型语言模型)** 等等, 甚至**任何自定义模型**。这是因为caflops支持基于Pytorch的```torch.nn.function.*```实现的计算操作。同时该工具支持打印模型各子模块的FLOPS、参数计算值和比例，方便用户了解模型各部分的性能消耗情况。

对于大模型，```calflops```相比其他工具可以更方便计算FLOPs，通过```calflops.calculate_flops()```您只需要通过参数```transformers_tokenizer```传递需要计算的transformer模型相应的```tokenizer```，它将自动帮助您构建```input_shape```模型输入。或者，您还可以通过``` args```， ```kwargs ```处理需要具有多个输入的模型，例如bert模型的输入需要```input_ids```, ```attention_mask```等多个字段。详细信息请参见下面```calflops.calculate_flops()```的api。

另外，这个包的实现过程受到[ptflops](https://github.com/sovrasov/flops-counter.pytorch)和[deepspeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed)库实现的启发，他们也都是非常好的工作。同时，calflops包也在他们基础上改进了一些方面(更简单的使用，更多的模型支持)，详细可以使用```pip install calflops```体验一下。


## 安装最新的版本
#### From PyPI:

```python
pip install calflops
```

同时你也可以从pypi calflops官方网址: https://pypi.org/project/calflops/ 
 上下载最新版本的whl文件 `calflops-*-py3-none-any.whl` 到本地进行离线安装：

```python
pip install calflops-*-py3-none-any.whl
```
## 如何使用calflops
### 举个例子
### CNN Model

如果模型的输入只有一个参数，你只需要通过对传入参数```input_shape```设置参数的大小即可，calflops会根据设定维度自动生成一个随机值作为模型的输入进行计算FLOPs。

```python
from calflops import calculate_flops
from torchvision import models

model = models.alexnet()
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M 
```

如果需要计算FLOPs的模型有多个输入，你也只需要通过传入参数 ```args``` 或 ```kargs```进行构造, 具体可以见下面Tranformer Model给出的例子。

### Transformer Model 

相比CNN Model，Transformer Model如果想使用参数 ```input_shape``` 指定输入数据的大小自动生成输入数据时额外还需要将其对应的```tokenizer```通过参数```transformer_tokenizer```进行传入，当然这种方式相比下面通过```kwargs```传入已构造输入数据方式更方便。

``` python
# Transformers Model, such as bert.
from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

batch_size = 1
max_seq_length = 128
model_name = "hfl/chinese-roberta-wwm-ext/"
model_save = "../pretrain_models/" + model_name
model = AutoModel.from_pretrained(model_save)
tokenizer = AutoTokenizer.from_pretrained(model_save)

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size,max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#Bert(hfl/chinese-roberta-wwm-ext) FLOPs:67.1 GFLOPS   MACs:33.52 GMACs   Params:102.27 M 
```

如果希望使用自己生成的特定数据来计算FLOPs，可以使用参数```args```或```kwargs```，这种情况参数```input_shape```不能再传入值。下面给出一个例子，可以看出没有通过```transformer_tokenizer```方便。

``` python
# Transformers Model, such as bert.
from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

batch_size = 1
max_seq_length = 128
model_name = "hfl/chinese-roberta-wwm-ext/"
model_save = "/code/yexiaoju/generate_tags/models/pretrain_models/" + model_name
model = AutoModel.from_pretrained(model_save)
tokenizer = AutoTokenizer.from_pretrained(model_save)

text = ""
inputs = tokenizer(text,
                   add_special_tokens=True, 
                   return_attention_mask=True,
                   padding=True,
                   truncation="longest_first",
                   max_length=max_seq_length)

if len(inputs["input_ids"]) < max_seq_length:
    apply_num = max_seq_length-len(inputs["input_ids"])
    inputs["input_ids"].extend([0]*apply_num)
    inputs["token_type_ids"].extend([0]*apply_num)
    inputs["attention_mask"].extend([0]*apply_num)
    
inputs["input_ids"] = torch.tensor([inputs["input_ids"]])
inputs["token_type_ids"] = torch.tensor([inputs["token_type_ids"]])
inputs["attention_mask"] = torch.tensor([inputs["attention_mask"]])

flops, macs, params = calculate_flops(model=model,
                                      kwargs = inputs,
                                      print_results=False)
print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#Bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.36 GFLOPS   MACs:11.17 GMACs   Params:102.27 M 
```


### Large Language Model

请注意，传入参数```transfromer_tokenizer```与大模型的tokenzier必须是一致匹配的。


``` python
#Large Languase Model, such as llama2-7b.
from calflops import calculate_flops
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
print("Llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#Llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B 
```

### 显示每个子模块的FLOPs, mac, Params

calflops提供了更详细的显示模型FLOPs计算信息。通过设置参数```print_result=True```，默认为True。如下图所示，在终端或jupyter界面打印模型的FLOPs。

![print_results](https://github.com/MrYxJ/calculate-flops.pytorch/blob/main/screenshot/alxnet_print_result.png?raw=true)

同时，通过设置参数```print_detailed =True```，默认为True。 calflops支持显示整个模型各子模块中FLOPs、NACs和Parameter的计算结果和占比的比例，这可以直接查看整个模型哪部分的消耗计算量最大，方便后续性能的优化。

![print_detailed](https://github.com/MrYxJ/calculate-flops.pytorch/blob/main/screenshot/alxnet_print_detailed.png?raw=true)

### 更多使用介绍

<details>
<summary> 如何使输出格式更优雅 </summary>
您可以使用参数```output_as_string```, ```output_precision```, ```output_unit```来确定输出数据的格式是value还是string，如果是string，则保留多少位精度和值的单位，例如FLOPs的单位是“TFLOPs”或“GFLOPs”，“MFLOPs”。

</details>

<details>
<summary> 如何处理有多个输入的模型 </summary>
calflops支持具有多个输入的模型，你只需使用参数args或kwargs进行构造，即可将多个输入作为模型推理的传入。
</details>

<details>
<summary> 如何让计算FLOPS的结果包括模型的正向和反向传播
 </summary>
你可以使用参数include_backPropagation来选择FLOPs结果的计算是否包含模型反向传播的过程，默认缺省值为False，即FLOPs只包含模型前向传播的过程。

此外，参数compute_bp_factor用于确定向后传播的计算次数与向前传播的计算次数相同。默认值缺省值是2.0，根据技术报告：https://epochai.org/blog/backward-forward-FLOP-ratio
</details>

<details>
<summary> 如何仅计算部分模型模块的FLOPs </summary>
你可以通过参数ignore_modules可以选择在计算FLOPs时忽略model中的哪些模块。默认为[]，即在计算结果包括模型的所有模块。
</details>

<details>
<summary> 如何计算LLM中生成函数(model.generate())的FLOPs </summary>
你只需要将“generate”赋值给参数forward_mode。
</details>

### **Api** of the **calflops**

<details>
<summary> calflops.calculate_flops() </summary>

``` python
from calflops import calculate_flops

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
</details>

<details>
<summary> calflops.generate_transformer_input()</summary>

``` python
def generate_transformer_input(model_tokenizer, input_shape, device):
    """Automatically generates data in the form of transformes model input format.
    
    Args:
        input_shape (tuple):transformers model input shape: (batch_size, seq_len).
        tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

    Returns:
        dict: data format of transformers model input, it is a dict which contain 'input_ids', 'attention_mask', 'token_type_ids' etc.
    """
```
</details>


</details>

## Citation
if calflops was useful for your paper or tech report, please cite me:
```
@online{calflops,
  author = {xiaoju ye},
  title = {calflops: a FLOPs and Params calculate tool for neural networks in pytorch framework},
  year = 2023,
  url = {https://github.com/MrYxJ/calculate-flops.pytorch},
}
```

## 常见模型的FLOPs

### Large Language Model
Input data format: batch_size=1, seq_len=128

- fwd FLOPs: The FLOPs of the model forward propagation

- bwd + fwd FLOPs: The FLOPs of model forward and backward propagation

另外注意这里fwd + bwd 没有包括模型参数激活的计算损耗，如果包括的对fwd的结果乘4即可。根据论文：https://arxiv.org/pdf/2205.05198.pdf

Model         | Input Shape | Params(B)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs(G) | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---         |---       |---        |--- 
bloom-1b7     |(1,128)      | 1.72B    | 1722408960  | 310.92     | 155.42   | 932.76    | 466.27
bloom-7b1     |(1,128)      | 7.07B    | 7069016064  | 1550.39    | 775.11   | 4651.18   | 2325.32
baichuan-7B   |(1,128)      | 7B       | 7000559616  | 1733.62    | 866.78   | 5200.85   | 2600.33
chatglm-6b    |(1,128)      | 6.17B    | 6173286400  | 1587.66    | 793.75   | 4762.97   | 2381.24
chatglm2-6b   |(1,128)      | 6.24B    | 6243584000  | 1537.68    | 768.8    | 4613.03   | 2306.4 
Qwen-7B       |(1,128)      | 7.72B    | 7721324544  | 1825.83    | 912.88   | 5477.48   | 2738.65
llama-7b      |(1,128)      | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550
llama2-7b     |(1,128)      | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550   
llama2-7b-chat |(1,128)     | 6.74B    | 6738415616  | 1700.06    | 850      | 5100.19   | 2550
chinese-llama-7b | (1,128)  | 6.89B    | 6885486592  | 1718.89    | 859.41   |5156.67   | 2578.24
chinese-llama-plus-7b| (1,128) | 6.89B | 6885486592  | 1718.89    | 859.41   |5156.67   | 2578.24
moss-moon-003-sft |(1,128) | 16.72B  | 16717980160 |  4124.93    | 2062.39  |  12374.8  | 6187.17

从上表中我们可以得出一些简单而有趣的发现:
- chatglm2-6b在相同比例的模型中，模型参数更小，FLOPs也更小，在速度性能上具有一定的优势。
- llama1-7b、llama2-7b和llama2-7b-chat模型参数一点没变、FLOPs也保持一致。符合[meta在其llama2报告](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)中描述的llama2-7b的模型结构没有改变，主要区别是训练数据token的增加。
- 类似的从表中可以看出，chinese-llama-7b和chinese-llama-plus-7b数据也符合[cui的报告](https://arxiv.org/pdf/2304.08177v1.pdf)，只是增加了更多的中文数据token进行训练，模型没有改变。
- ......

更多的模型FLOPs将陆续更新，参见github
[calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch)

### Bert

Input data format: batch_size=1, seq_len=128

Model         | Input Shape | Params(M)|Params(Total)| fwd FLOPs(G) | fwd MACs(G) | fwd + bwd FLOPs() | fwd + bwd MACs(G)  | 
---           |---          |---       |---          |---        |---       |---     |---
hfl/chinese-roberta-wwm-ext | (1,128)| 102.27M | 102267648 |       67.1  |    33.52  |  201.3 | 100.57
......

你可以使用calflops来计算基于bert的更多不同模型，期待你更新在此表中。


## Benchmark
### [torchvision](https://pytorch.org/docs/1.0.0/torchvision/models.html)

Input data format: batch_size = 1, actually input_shape = (1, 3, 224, 224)

注:表中FLOPs仅考虑模型正向传播的计算，**Total**为不含单位缩写的总数值表示。

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

感谢 @[zigangzhao-ai](https://github.com/zigangzhao-ai) 帮忙使用 ```calflops``` 去统计表 torchvision的结果. 

你也可以将calflops计算FLOPs的结果与其他优秀的工具计算结果进行比较
: [ptflops readme.md](https://github.com/sovrasov/flops-counter.pytorch/).


## Concact Author

Author: [MrYXJ](https://github.com/MrYxJ/)

Mail: yxj2017@gmail.com
