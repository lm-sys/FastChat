import transformers
import os
import torch

CKPT_PATH = "/home/gcpuser/ckpt"

CONTEXT = ("A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
           "Human: Hello! What can you do?\n"
           "AI: As an AI assistant, I can answer questions and chat with you.\n")
#           "Human: What is the name of the tallest mountain in the world?\n")

class DialoGPT:
    def __init__(
        self,
        model,
        tokenizer,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = CONTEXT

    def __call__(self, inputs: str) -> str:
        # wrapped_inputs = 'Human: ' + inputs + self.tokenizer.eos_token
        wrapped_inputs = "Human: " + inputs + "\n"
        self.history += wrapped_inputs
        input_ids = self.tokenizer(self.history, return_tensors="pt").input_ids.to(self.device)
        # input_ids = input_ids[:, -200:]
        # 
        output_ids = self.model.generate(
            input_ids, do_sample=True, temperature=0.9, max_length=405
        )
        generated_text = self.tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        self.history += generated_text
        return generated_text

    def run(self):
        while True:
            user_input = input("Human: ")
            print(self(user_input))

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(CKPT_PATH).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CKPT_PATH,
        padding_side="right",
        use_fast=False,
    )

    bot = DialoGPT(model, tokenizer, device)
    bot.run()
