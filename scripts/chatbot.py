import transformers
import os
import torch

CKPT_PATH = "/home/gcpuser/ckpt"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

CONTEXT = ("A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
           "Human: Hello! What can you do?\n"
           "Assistant: As an AI assistant, I can answer questions and chat with you.\n")
#           "Human: What is the name of the tallest mountain in the world?\n")

class DialoGPT:
    def __init__(
        self
    ):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(CKPT_PATH).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            CKPT_PATH,
            padding_side="right",
            use_fast=False,
        )
        self.history_ids = None

    def __call__(self, inputs: str) -> str:
        input_ids = self.tokenizer('Human: '+inputs, return_tensors="pt").input_ids.to(device)
        if self.history_ids is None:
            context_ids = self.tokenizer(CONTEXT, return_tensors="pt").input_ids.to(device)
            input_ids = torch.cat([context_ids, input_ids], dim=-1)
        else:
            input_ids = torch.cat([self.history_ids, input_ids], dim=-1)
        self.history_ids = self.model.generate(
            input_ids, do_sample=True, temperature=0.9, max_length=200
        )
        generated_text = self.tokenizer.decode(self.history_ids[:, input_ids.shape[-1]:][0])
        return generated_text

    def run(self):
        while True:
            user_input = input("Human: ")
            print(self(user_input))


if __name__ == "__main__":
    bot = DialoGPT()
    bot.run()
