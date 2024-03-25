
# Integrating Guidance Model with Fast-Chat OpenAI API Server

This guide provides instructions on how to start a Fast-Chat OpenAI API server and utilize the Guidance model for generating responses.

## Step 1: Start the Fast-Chat OpenAI API Server

Refer to the `README.md` file provided with the Fast-Chat package for detailed instructions on starting the server. This step involves initializing the server which will be running locally, typically accessible at `http://127.0.0.1:8000/v1`.

## Step 2: Utilize the Guidance Model

Once the Fast-Chat server is running, you can use the Guidance model to generate responses. Below is an example Python script demonstrating how to integrate and use the Guidance model with the Fast-Chat API:

```python
# Import necessary modules
from fastchat.guidance.model import LiteLLMCompletion
import litellm
from guidance import gen

# Server URL and model details
base_url = "http://127.0.0.1:8000/v1"
model = "vicuna-7b-v1.5"

# Initialize the LiteLLMCompletion instance
llm = LiteLLMCompletion(api_base=base_url, model=model, api_key="empty", custom_llm_provider="openai", echo=False)

# Generate and append text or responses using the model
response = llm + f'USER: Could you please tell me a joke?  ASSISTANT: ' + gen(stop=".") + ". USER: Another one? ASSISTANT: " + \
    gen(stop=".", max_tokens=100)

# Print the generated conversation
print(response.__str__())
```

For additional options and detailed syntax, please refer to the [Guidance documentation](https://github.com/guidance-ai/guidance).

---

This guide should help you set up and utilize the Guidance integration with Fast-Chat.