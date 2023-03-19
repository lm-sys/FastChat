import transformers
import os
import torch

CKPT_PATH = "/home/gcpuser/ckpt"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    def __call__(self, inputs: str) -> str:
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(device)
        generated_ids = self.model.to(device).generate(
            input_ids, do_sample=True, temperature=0.9, max_length=200
        )
        generated_text = self.tokenizer.decode(generated_ids[0])
        return generated_text

    def run(self):
        while True:
            user_input = input("User: ")
            print("Bot:", self(user_input))


if __name__ == "__main__":
    bot = DialoGPT()
    bot.run()
