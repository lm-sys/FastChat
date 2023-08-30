# Chatgpt

You can add access to chatgpt(openai/azure) in fastchat

# How

1. Add environment variables.  
```shell
# GPT-3.5-turbo
export OPENAI_API_BASE_GPT35="xxx"
export OPENAI_API_KEY_GPT35="xxx"
export OPENAI_API_VERSION_GPT35="xxx"
export OPENAI_API_TYPE_GPT35="xxx"
export OPENAI_ENGINE_GPT35="xxx"

# GPT-4
export OPENAI_API_BASE_GPT4="xxx"
export OPENAI_API_KEY_GPT4="xxx"
export OPENAI_API_VERSION_GPT4="xxx"
export OPENAI_API_TYPE_GPT4="xxx"
export OPENAI_ENGINE_GPT4="xxx"
```

2. Add the `--add-chatgpt` command when starting the web service.  
```shell
python3 -m fastchat.serve.gradio_web_server  --add-chatgpt
python3 -m fastchat.serve.gradio_web_server_multi  --add-chatgpt
```
