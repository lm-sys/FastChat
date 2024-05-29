项目clone自 https://github.com/lm-sys/FastChat

在[openai_api_server.py](fastchat%2Fserve%2Fopenai_api_server.py)中增加了tools(function)能力

可使用langgraph调用
```python
prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = LocalChat()
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

```
## 支持模型
Qwen

## 声明

只在Qwen1.5 72B 模型上进行过测试，其他模型不保证能够正常使用。

基于此 未向 FastChat 提交PR
