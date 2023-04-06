from typing import Any, Dict, List, Tuple

import torch


class BaseModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate_stream(self, input_text: str, params: Dict[str, Any], **kwargs):
        raise NotImplementedError(
            "generate_stream() should be implemented for the model type you want to support.")


class LLamaWrapper(BaseModelWrapper):
    """Wrapper for LLaMA https://huggingface.co/docs/transformers/main/en/model_doc/llama"""

    def generate_stream(self, input_text: str, params: Dict[str, Any], **kwargs):
        device = str(params.get("device", "cuda"))
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        context_len = int(params.get("context_len", 2048))
        stream_interval = int(params.get("stream_interval", 2))
        stop_str = params.get("stop", None)

        input_ids = self.tokenizer(input_text).input_ids
        output_ids = list(input_ids)
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = self.model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == self.tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, len(input_text))
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

        del past_key_values


class ChatGlmWrapper(BaseModelWrapper):
    """Wrapper for ChatGLM-6B https://huggingface.co/THUDM/chatglm-6b/tree/main"""

    def generate_stream(self, input_text: str, params: Dict[str, Any], **kwargs):
        # The following default values follow kwarg default values of chatglm-6b's `stream_chat`
        # https://huggingface.co/THUDM/chatglm-6b/blob/1b54948bb28de5258b55b893e193c3046a0b0484/modeling_chatglm.py#L1116
        history = params.get("history", None)
        max_length = int(params.get("context_len", 2048))
        do_sample = bool(params.get("do_sample", True))
        top_p = float(params.get("top_p", 0.7))
        temperature = float(params.get("temperature", 0.95))
        logits_processor = params.get("logits_processor", None)

        response_generator = self.model.stream_chat(
            self.tokenizer, input_text,
            history=history, max_length=max_length, temperature=temperature,
            do_sample=do_sample, top_p=top_p, logits_processor=logits_processor, **kwargs)

        # TODO (jiaodong): Support chat history as context for existing models and ChatGLM 
        for response, new_history in response_generator:
            yield response
