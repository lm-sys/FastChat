# LM-Arena demo for checklist assistant

**TLDR:** [Install](#install) then run:

- In 1 terminal:

`python3 -m fastchat.serve.controller`

- Create a JSON configuration file `api_endpoint.json` with the api endpoints of the models you want to serve. **Note, ask me or Ihsan the API keys by email**. For example:

```
{
    "Checklist-GPT-4-0125-Preview": {
        "model_name": "Checklist-GPT-4-0125-Preview",
        "api_type": "openai",
        "azure_api_version": "2024-02-01",
        "api_base": "https://checklist.openai.azure.com/",
        "api_key": "",
        "anony_only": false,
        "recommended_config": {
        "temperature": 0.7,
        "top_p": 1.0
        },
        "text-arena": true,
        "vision-arena": false
    },
    "Checklist-GPT-o1": {
        "model_name": "Checklist-GPT-o1",
        "api_type": "openai_o1",
        "azure_api_version": "2024-02-01",
        "api_base": "https://checklist.openai.azure.com/",
        "api_key": "",
        "anony_only": false,
        "recommended_config": {
        "temperature": 0.7,
        "top_p": 1.0
        },
        "text-arena": true,
        "vision-arena": false
    }
}
```

- Add the sampling rate in the `SAMPLING_WEIGHTS` dictionary in `fastchat/serve/gradio_block_arena_anony.py` .
- In 2nd terminal:

`python3 -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json`

## Contents

- [Install](#install)
- [Serving with Web GUI](#serving-with-web-gui)
- [API](#api)
- [Citation](#citation)

## Install

### Method 1: With pip

```bash
pip3 install "fschat[model_worker,webui]"
```

### Method 2: From source

1. Clone this repository and navigate to the FastChat folder

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

If you are running on Mac:

```bash
brew install rust cmake
```

2. Install Package

```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```

## Serving with Web GUI

`<a href="https://lmarena.ai"><img src="assets/screenshot_gui.png" width="70%">``</a>`

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. You can learn more about the architecture [here](docs/server_arch.md).

Here are the commands to follow in your terminal:

#### Launch the controller

```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

#### Launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

To ensure that your model worker is connected to your controller properly, send a test message using the following command:

```bash
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
```

You will see a short output.

#### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now.
If the models do not show up, try to reboot the gradio web server.

## Launch Chatbot Arena (side-by-side battle UI)

Currently, Chatbot Arena is powered by FastChat. Here is how you can launch an instance of Chatbot Arena locally.

FastChat supports popular API-based models such as OpenAI, Anthropic, Gemini, Mistral and more. To add a custom API, please refer to the model support [doc](./docs/model_support.md). Below we take OpenAI models as an example.

Create a JSON configuration file `api_endpoint.json` with the api endpoints of the models you want to serve, for example:

```
{
    "gpt-4o-2024-05-13": {
        "model_name": "gpt-4o-2024-05-13",
        "api_base": "https://api.openai.com/v1",
        "api_type": "openai",
        "api_key": [Insert API Key],
        "anony_only": false
    }
}
```

For Anthropic models, specify `"api_type": "anthropic_message"` with your Anthropic key. Similarly, for gemini model, specify `"api_type": "gemini"`. More details can be found in [api_provider.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/api_provider.py).

To serve your own model using local gpus, follow the instructions in [Serving with Web GUI](#serving-with-web-gui).

Now you're ready to launch the server:

```
python3 -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
```

#### (Optional): Advanced Features, Scalability, Third Party UI

- You can register multiple model workers to a single controller, which can be used for serving a single model with higher throughput or serving multiple models at the same time. When doing so, please allocate different GPUs and ports for different model workers.

```
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
```

- You can also launch a multi-tab gradio server, which includes the Chatbot Arena tabs.

```bash
python3 -m fastchat.serve.gradio_web_server_multi
```

- The default model worker based on huggingface/transformers has great compatibility but can be slow. If you want high-throughput batched serving, you can try [vLLM integration](docs/vllm_integration.md).
- If you want to host it on your own UI or third party UI, see [Third Party UI](docs/third_party_ui.md).

## API

### OpenAI-Compatible RESTful APIs & SDK

FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs.
The FastChat server is compatible with both [openai-python](https://github.com/openai/openai-python) library and cURL commands.
The REST API is capable of being executed from Google Colab free tier, as demonstrated in the [FastChat_API_GoogleColab.ipynb](https://github.com/lm-sys/FastChat/blob/main/playground/FastChat_API_GoogleColab.ipynb) notebook, available in our repository.
See [docs/openai_api.md](docs/openai_api.md).

### Hugging Face Generation APIs

See [fastchat/serve/huggingface_api.py](fastchat/serve/huggingface_api.py).

### LangChain Integration

See [docs/langchain_integration](docs/langchain_integration.md).

## Citation

The code (training, serving, and evaluation) in this repository is mostly developed for or derived from the paper below.
Please cite it if you find the repository helpful.

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We are also planning to add more of our research to this repository.
