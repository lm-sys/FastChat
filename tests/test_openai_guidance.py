from fastchat.guidance.model import LiteLLMCompletion
import litellm
from guidance import gen

base_url = "http://127.0.0.1:8000/v1"
model = "vicuna-7b-v1.5"
llm = LiteLLMCompletion(
    api_base=base_url,
    model=model,
    api_key="empty",
    custom_llm_provider="openai",
    echo=False,
)
# append text or generations to the model
res = (
    llm
    + f"USER: Could you please tell me a joke?  ASSISTANT: "
    + gen(stop=".")
    + ". USER: Another one? ASSISTANT:"
    + gen(stop=".", max_tokens=100)
)
print(res.__str__())
