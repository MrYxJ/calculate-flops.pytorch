# calculate-flops.pytorch

```python
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
#bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.363 GFLOPS   MACs:11.174 GMACs   Params:102.268 M 

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
#llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.001 GMACs   Params:6.738 B 
```