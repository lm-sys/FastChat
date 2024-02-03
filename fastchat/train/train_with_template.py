# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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

from dataclasses import dataclass, field
import json
import math
import jsonlines
import pathlib
from multiprocessing import Pool
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def apply_prompt_template(sources, template_id, systems=None):
    conv = get_conversation_template(template_id)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if systems and systems[i]:
            conv.set_system_message(systems[i])
        prompt = conv.get_prompt()
        conversations.append(prompt)
    return conversations, conv


def tokenize_conversations(conversations, tokenizer):
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    return input_ids, targets


def get_prompt_separator(conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        if conv.sep2 is None:
            user_turn_separator = conv.roles[0] + ": "
        else:
            user_turn_separator = conv.sep2

        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.LLAMA2:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + " "

    elif conv.sep_style == SeparatorStyle.CHATML:
        if conv.sep2 is None:
            user_turn_separator = conv.sep + "\n"
        else:
            user_turn_separator = conv.sep2 + "\n"

        assistant_turn_separator = conv.roles[1] + "\n"

    return user_turn_separator, assistant_turn_separator


def mask_targets(conversations, targets, tokenizer, conv):
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if tokenizer.eos_token is None:
            cur_len = 0
        elif tokenizer.eos_token is not None and target[0] != tokenizer.bos_token_id:
            cur_len = 0
        elif tokenizer.eos_token is not None and target[0] == tokenizer.bos_token_id:
            cur_len = 1

        target[:cur_len] = IGNORE_TOKEN_ID
        user_turn_separator, assistant_turn_separator = get_prompt_separator(conv)
        turns = conversation.split(user_turn_separator)
        for i, turn in enumerate(turns):
            if (
                i < len(turns) - 1 and turn == ""
            ):  # Last turn is the user_turn_separator
                break

            if i != 0:
                turn = user_turn_separator + turn

            turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

            if assistant_turn_separator in turn:
                parts = turn.rsplit(assistant_turn_separator)
                parts[0] += assistant_turn_separator
            else:
                parts = [turn]

            instruction_len = len(
                tokenizer(parts[0], add_special_tokens=False).input_ids
            )

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return targets


def preprocess(
    sources, tokenizer: transformers.PreTrainedTokenizer, template_id, **kwargs
) -> Dict:
    systems = None if not kwargs else kwargs.get("systems", None)

    # If the data volume is small, process it directly in the main thread
    if len(sources) <= 1000:
        conversations, conv = apply_prompt_template(sources, template_id, systems)
        input_ids, targets = tokenize_conversations(conversations, tokenizer)
        targets = mask_targets(conversations, targets, tokenizer, conv)
    else:  # If the data volume is large, use multithreading for processing
        with Pool() as p:
            conversations, conv = p.apply_async(
                apply_prompt_template, (sources, template_id, systems)
            ).get()
            input_ids, targets = p.apply_async(
                tokenize_conversations, (conversations, tokenizer)
            ).get()
            targets = p.apply_async(
                mask_targets, (conversations, targets, tokenizer, conv)
            ).get()
            p.close()
            p.join()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template_id
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        systems = [example.get("system", "") for example in raw_data]
        sources = [example["conversations"] for example in raw_data]

        data_dict = preprocess(sources, tokenizer, template_id, systems=systems)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template_id
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_id = template_id

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(
            [self.raw_data[i]["conversations"]],
            self.tokenizer,
            self.template_id,
            systems=[self.raw_data[i].get("system", "")],
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    template_id,
    train_ratio=0.98,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_ratio = min(train_ratio, 1.0)
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    data_path = data_args.data_path
    if data_path.endswith(".json"):
        raw_data = json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, mode="r") as reader:
            raw_data = [item for item in reader]

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * train_ratio)
    train_indices = perm[:split]
    if train_ratio < 1:
        eval_indices = perm[split:]
    else:
        # if train_ratio==1, we use 5% of data as eval data, make sure trainer will not throw error when eval data is empty
        eval_indices = perm[-int(len(perm) * 0.05) :]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(
        train_raw_data, tokenizer=tokenizer, template_id=template_id
    )
    eval_dataset = dataset_cls(
        eval_raw_data, tokenizer=tokenizer, template_id=template_id
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    # Tie the weights
    model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # NOTE: if the token_id exceed the vocab_size will cause failing in training process! we need add special config and resize the embedding size!
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print(f"tokens len: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    template_id = model_args.model_name_or_path
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        template_id=template_id,
        train_ratio=0.98,
        data_args=data_args,
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
