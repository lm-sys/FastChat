# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from typing import List

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="/share/zhanghuaao/model_hubs/llama/tokenizer_zh/")
    freeze: Optional[bool] = field(default=False)


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


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles.get(source[0]["from"], None) != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]: # skip 
                break
            if sentence["value"].strip() == "": # skip empty value
                break
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print('Genetationg total {} samples'.format(len(conversations)))

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
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
            target[cur_len:cur_len+instruction_len] = (
                IGNORE_TOKEN_ID)

            # rank0_print(tokenizer.decode(target))

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(f"WARNING: tokenization mismatch "
                            f"{cur_len} vs. {total_len}")
                

    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

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
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.valid_path)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset)


def train():
    import pdb;pdb.set_trace()
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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
