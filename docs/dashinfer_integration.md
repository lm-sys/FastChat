# dash-infer Integration
[DashInfer](https://github.com/modelscope/dash-infer) is a high-performance inference engine specifically optimized for CPU environments, delivering exceptional performance boosts for LLM inference tasks. It supports acceleration for a variety of models including Llama, Qwen, and ChatGLM, making it a versatile choice as a performant worker in FastChat. Notably, DashInfer exhibits significant performance enhancements on both Intel x64 and ARMv9 processors, catering to a wide spectrum of hardware platforms. Its efficient design and optimization techniques ensure rapid and accurate inference capabilities, making it an ideal solution for deploying large language models in resource-constrained environments or scenarios where CPU utilization is preferred over GPU acceleration.

## Instructions
1. Install dash-infer.
    ```
    pip install dashinfer
    ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the dash-infer worker (`fastchat.serve.dashinfer_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.
   ```
   python3 -m fastchat.serve.dashinfer_worker --model-path qwen/Qwen-7B-Chat --revision=master /path/to/dashinfer-model-generation-config.json
   ```
Here is an example:
   ```
   python3 -m fastchat.serve.dashinfer_worker --model-path qwen/Qwen-7B-Chat --revision=master dash-infer/examples/python/model_config/config_qwen_v10_7b.json
   ```

   If you use an already downloaded model, try to replace model-path with a local one and choose a conversation template via --conv-template option
   '''
   python3 -m fastchat.serve.dashinfer_worker --model-path ~/.cache/modelscope/hub/qwen/Qwen-7B-Chat --conv-template qwen-7b-chat /path/to/dashinfer-model-generation-config.json
   '''
   All avaliable conversation chat templates are listed at [fastchat/conversation.py](../fastchat/conversation.py)
