import torch
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download("ccyh123/Qwen-7B-Chat", revision='v1.0.0', cache_dir='/home/Userlist/madehua/model/Qwen')
model_dir = snapshot_download("ZhipuAI/chatglm2-6b", revision='v1.0.9', cache_dir='/home/Userlist/madehua/model/chatglm')
model_dir = snapshot_download("skyline2006/llama-7b", revision='v1.0.1', cache_dir='/home/Userlist/madehua/model/llama2')
model_dir = snapshot_download("baichuan-inc/Baichuan2-13B-Chat", revision='v1.0.2', cache_dir='/home/Userlist/madehua/model/Baichuan')
model_dir = snapshot_download("skyline2006/llama-13b", revision='v1.0.0', cache_dir='/home/Userlist/madehua/model/llama2')


