# FastChat for tool
项目clone自 https://github.com/lm-sys/FastChat

在[openai_api_server.py](fastchat%2Fserve%2Fopenai_api_server.py)中增加了tools(function)能力，可在其他需要调用工具的场景下使用。(如果对您有用，麻烦点个star🙇‍♂️)

如 langgraph调用
```python
from langchain_core.pydantic_v1 import SecretStr
from langchain_openai import ChatOpenAI

class LocalChat(ChatOpenAI):
    openai_api_base = "http://127.0.0.1:8002/v1"
    openai_api_key = SecretStr("123456")
    model_name = "Qwen1.5-72B-Chat-GPTQ-Int8"
    temperature = 0.0

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = LocalChat()
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

```
## 支持模型
Qwen(已测试)

## 使用方式
#### Launch the controller
```python
python3 -m fastchat.serve.controller
```
#### Launch the model worker(s)
```python
python3 -m fastchat.serve.model_worker --model-path /models/qwen/Qwen1.5-72B-Chat-GPTQ-Int8
```

#### Launch the API

```python
python -m fastchat.serve.openai_api_server_for_tool --controller-address http://127.0.0.1:21001
```
## 安装方式

Option1: 直接clone项目`git clone https://github.com/bluechanel/FastChat`，

其他步骤 https://github.com/lm-sys/FastChat?tab=readme-ov-file#install

Option2: `python -m build` 打包whl包后pip安装

## 声明

只在Qwen1.5 72B 模型上进行过测试，其他模型不保证能够正常使用。

基于此 未向 FastChat 提交PR
