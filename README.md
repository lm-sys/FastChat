# FastChat
An open platform for training, serving, and evaluating large language model based chatbots.

## Release
- 🔥 We released **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality**. Checkout the blog [post]() and [demo]().

[A GIF HERE].

Join our [Discord]() server and follow our [Twitter]() to get the latest updates.

## Contents
- [Install](#install)
- [Serving](#serving)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)

## Install

### Method 1: From Source
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install -e .

# Install the latest main branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers
```

## Serving

### Command Line Interface
```
python3 -m fastchat.serve.cli --model facebook/opt-1.3b
```

### Web UI
```
# Launch a controller
python3 -m fastchat.serve.controller

# Launch a model worker
python3 -m fastchat.serve.model_worker --model facebook/opt-1.3b

# Send a test message
python3 -m fastchat.serve.test_message

# Luanch a gradio web server.
python3 -m fastchat.serve.gradio_web_server

# You can open your brower and chat with a model now.
```

## Fine-tuning


## Evaluation

