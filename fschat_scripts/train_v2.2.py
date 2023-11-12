# 1. flash attention2
# 2. 可加验证集
# 3. 动态batch

import copy, math
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import random
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# from fastchat.train.llama_flash_attn2_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
from fastchat.conversation import get_default_conv_template, SeparatorStyle
from typing import List

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TEMPLATE = get_default_conv_template("vicuna").copy()
SYSTEM = TEMPLATE.system

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="/share/zhanghuaao/models/tokenizer_zh/")
    freeze: Optional[bool] = field(default=False)
    template: Optional[str] = 'vicuna'


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the training data."})
    valid_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def is_match(lista, listb):
    for a, b in zip(lista, listb):
        if a != b:
            return False
    return True

def mask_target(conv, input_id, tokenizer):
    sep = TEMPLATE.sep + TEMPLATE.roles[1] + ": "
    rounds = conv.split(TEMPLATE.sep2)
    cur_len = 1
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        if len(parts) != 2:
            rank0_print('len(parts)={}'.format(len(parts)))
            break
        parts[0] += sep
        round_len = len(tokenizer(rou).input_ids)
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2
        # rank0_print(tokenizer.decode(target[cur_len:cur_len+instruction_len]))
        input_id[cur_len:cur_len+instruction_len] = (
            IGNORE_TOKEN_ID)

        # rank0_print(tokenizer.decode(target))

        cur_len += round_len
    input_id[cur_len:] = IGNORE_TOKEN_ID
    return input_id


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    roles = {"human": TEMPLATE.roles[0], "gpt": TEMPLATE.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        
        if source[0]['from'] == 'system':  # 加入system角色
            TEMPLATE.system = source[0]['value']
            source = source[1:]
        else:
            TEMPLATE.system = SYSTEM

        if roles.get(source[0]["from"], None) != TEMPLATE.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        TEMPLATE.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != TEMPLATE.roles[j % 2]: # skip 
                break
            if sentence["value"].strip() == "": # skip empty value
                break
            TEMPLATE.append_message(role, sentence["value"])
        conversations.append(TEMPLATE.get_prompt())
    # import ipdb;ipdb.set_trace()
    from tqdm import tqdm
    batched_input_ids = [[]]
    targets = [[]]
    random.shuffle(conversations)
    # 策略：
    # 1. 多条能塞下就塞下
    # 2. 单条塞不下就拆分(TODO 处理报错)
    for c in tqdm(conversations):
        input_id = tokenizer(c).input_ids
        target = mask_target(c, torch.tensor(input_id), tokenizer).tolist()
        if len(input_id) + len(batched_input_ids[-1]) <= tokenizer.model_max_length:
            batched_input_ids[-1] += input_id
            targets[-1] += target
        else:
            if len(input_id) <= tokenizer.model_max_length:
                batched_input_ids.append(input_id)
                targets.append(target)
            else:
                for block in range(math.ceil(len(input_id) / tokenizer.model_max_length)):
                    batched_input_ids.append(input_id[block * tokenizer.model_max_length: (block+1) * tokenizer.model_max_length])
                    targets.append(target[block * tokenizer.model_max_length: (block+1) * tokenizer.model_max_length])

    batched_input_ids = [data for data in batched_input_ids if len(data) > 32]
    targets = [data for data in targets if len(data) > 32]
                # import ipdb;ipdb.set_trace()
                # batched_input_ids.append(input_id[:tokenizer.model_max_length])
                # targets.append(target[:tokenizer.model_max_length])

    max_len = tokenizer.model_max_length

    for idx, input_id in enumerate(batched_input_ids):
        batched_input_ids[idx] += [tokenizer.pad_token_id] * (max_len - len(input_id))
        targets[idx] += [tokenizer.pad_token_id] * (max_len - len(targets[idx]))

    input_ids = torch.tensor(batched_input_ids)
    targets = torch.tensor(targets)
    
    assert TEMPLATE.sep_style == SeparatorStyle.TWO

    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = []
        for path in data_path.split('@@'):

            print(f"Loading data : {path}")
            data_dict = json.load(open(path, "r"))
            list_data_dict.extend(data_dict)

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: List[str],
                 tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        
        list_data_dict = []
        for path in data_path.split('@@'):

            print(f"Loading data : {path}")
            data_dict = json.load(open(path, "r"))
            list_data_dict.extend(data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess([e["conversations"] for e in sources],
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             attention_mask=data_dict["attention_mask"][0])
        return data_dict


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_args.lazy_preprocess = 'False'
    print('auto set lazy_preprocess to False')
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.valid_path)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset)


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    import pdb;pdb.set_trace()
    # 加载template
    global TEMPLATE, SYSTEM
    TEMPLATE = get_default_conv_template(model_args.template).copy()
    SYSTEM = TEMPLATE.system

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # import ipdb;ipdb.set_trace()
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        max_position_embeddings=training_args.model_max_length
    )
    model.config.use_cache=False
    model.resize_token_embeddings(len(tokenizer))
    if model_args.freeze:
        layers = model.config.num_hidden_layers
        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'lm_head' in name:
                param.requires_grad = True
                print('Updating params :{}'.format(name))
            if str(layers) in name or str(layers - 1) in name:
                print('Updating params :{}'.format(name))
                param.requires_grad = True
            

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
