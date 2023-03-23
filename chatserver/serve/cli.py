import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from chatserver.conversation import default_conversation
from chatserver.utils import disable_torch_init


@torch.inference_mode()
def main(args):
    model_name = args.model_name
    num_gpus = args.num_gpus

    # Model
    disable_torch_init()
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
       model_name, torch_dtype=torch.float16, **kwargs)

    if num_gpus == 1:
        model.cuda()

    # Chat
    conv = default_conversation.copy()
    while True:
        inp = input(f"{conv.roles[0]}: ")
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))
        
        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        print(f"{conv.roles[1]}: {outputs}")
        conv.append_message(conv.roles[1], outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    main(args)
