import os
import torch
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel,AutoConfig
from transformers.generation.utils import GenerationConfig
MODEL_PATH = "/cpfs/29cd2992fe666f2a/user/zhangge/xw/Humpback-CH/checkpoint/internlm-20b"

def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_PATH
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model, tokenizer = init_model()
    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBaichuan 2：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()