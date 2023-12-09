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

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from dataclasses import dataclass, field
import os
import json
import math
import jsonlines
import pathlib
from multiprocessing import Pool
import random
from typing import Dict, Optional, List, Tuple

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import get_last_checkpoint

from fastchat.conversation import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    template_name: Optional[str] = field(default="vicuna")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
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


def trainer_save_model_safe(trainer: transformers.Trainer, output_dir=None):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model(output_dir=output_dir)


def save_deepspeed_model(trainer, tokenizer, output_dir=None):
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

    if output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-final")
        # checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
    else:
        checkpoint_dir = "./checkpoint-final"

    try:
        trainer.save_model(checkpoint_dir)
    except Exception as e:
        print(f"saving model with error {e}!")
        trainer.deepspeed.save_checkpoint(checkpoint_dir)

        fp16_model = load_state_dict_from_zero_checkpoint(
            trainer.model, checkpoint_dir
        ).half()
        fp16_model.config.torch_dtype = "float16"
        fp16_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)


class PreprocessorRaw(object):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, template_name="vicuna_v1.1"
    ):
        self.tokenizer = tokenizer
        self.conv = get_conv_template(template_name)

        self.roles = {"human": self.conv.roles[0], "gpt": self.conv.roles[1]}
        self.turn_sep = self.conv.sep + self.roles.get("gpt") + self.conv.role_sep

    def apply_prompt_template(self, sources, systems=None) -> List[str]:
        conversations = []
        for i, source in enumerate(sources):
            if source[0]["from"] == "gpt":
                source.insert(
                    0,
                    {
                        "from": "human",
                        "value": random.choice(
                            ["ping", "come", "begin", "start", "", "default"]
                        ),
                    },
                )
            self.conv.messages = []
            for j, sentence in enumerate(source):
                role = self.roles.get(sentence["from"])
                assert role == self.conv.roles[j % 2], f"{i}"
                self.conv.append_message(role, sentence["value"])
            if systems and systems[i]:
                self.conv.set_system_message(systems[i])
            prompt = self.conv.get_prompt()
            conversations.append(prompt)
        return conversations

    def tokenize_conversations(self, conversations) -> Tuple[List, List]:
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        return input_ids, targets

    def mask_targets(self, conversations, targets) -> List[int]:
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            turns = conversation.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(self.tokenizer(turn).input_ids)

                parts = turn.split(self.turn_sep)
                if len(parts) != 2:
                    break
                parts[0] += self.turn_sep

                # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                self.__debug(target)

            if cur_len < self.tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        return targets

    def __debug(self, target):
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
        rank0_print(self.tokenizer.decode(z))

    def preprocess(self, sources, **kwargs) -> Dict:
        systems = kwargs.get("systems", None) if kwargs else None

        # If the data volume is small, process it directly in the main thread
        if len(sources) <= 1000:
            conversations = self.apply_prompt_template(sources, systems)
            input_ids, targets = self.tokenize_conversations(conversations)
            targets = self.mask_targets(conversations, targets)
        else:  # If the data volume is large, use multithreading for processing
            with Pool() as p:
                conversations = p.apply_async(
                    self.apply_prompt_template, (sources, systems)
                ).get()
                input_ids, targets = p.apply_async(
                    self.tokenize_conversations, (conversations)
                ).get()
                targets = p.apply_async(
                    self.mask_targets, (conversations, targets)
                ).get()
                p.close()
                p.join()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class Preprocessor(object):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, template_name="vicuna_v1.1"
    ):
        self.tokenizer = tokenizer
        self.conv = get_conv_template(template_name)

        self.roles = {"human": self.conv.roles[0], "gpt": self.conv.roles[1]}

    def apply_mask(self, sources, systems=None) -> List[str]:
        batch_input_ids = []
        batch_targets = []
        for i, source in enumerate(sources):
            if source[0]["from"] == "gpt":
                source.insert(
                    0,
                    {
                        "from": "human",
                        "value": random.choice(
                            ["ping", "come", "begin", "start", "", "default"]
                        ),
                    },
                )
            self.conv.messages = []
            for j, sentence in enumerate(source):
                role = self.roles.get(sentence["from"])
                assert role == self.conv.roles[j % 2], f"{i}"
                self.conv.append_message(role, sentence["value"])
            if systems and systems[i]:
                self.conv.set_system_message(systems[i])

            input_ids, targets = self.conv.get_prompt_token_ids(self.tokenizer)

            input_ids = input_ids[: self.tokenizer.model_max_length]
            targets = targets[: self.tokenizer.model_max_length]
            input_ids += [self.tokenizer.pad_token_id] * (
                self.tokenizer.model_max_length - len(input_ids)
            )
            targets += [IGNORE_TOKEN_ID] * (
                self.tokenizer.model_max_length - len(targets)
            )

            batch_input_ids.append(input_ids)
            batch_targets.append(targets)

            if False:
                self.__debug(torch.IntTensor(input_ids), torch.IntTensor(targets))

        return torch.IntTensor(batch_input_ids), torch.IntTensor(batch_targets)

    def __debug(self, input_ids, target):
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
        rank0_print(self.tokenizer.decode(input_ids))
        rank0_print(self.tokenizer.decode(z))

    def preprocess(self, sources, **kwargs) -> Dict:
        systems = kwargs.get("systems", None) if kwargs else None

        # If the data volume is small, process it directly in the main thread
        if len(sources) <= 1000:
            input_ids, targets = self.apply_mask(sources, systems)
        else:  # If the data volume is large, use multithreading for processing
            with Pool() as p:
                input_ids, targets = p.apply_async(
                    self.apply_mask, (sources, systems)
                ).get()
                p.close()
                p.join()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, preprocessor: Preprocessor):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        systems = [example.get("system", "") for example in raw_data]
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocessor.preprocess(sources, systems=systems)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
            )
        except Exception as e:
            print(f"Getting item error {e}")
            # return a dummy sample
            return dict(
                input_ids=torch.tensor([]),
                labels=torch.tensor([]),
                attention_mask=torch.tensor([]),
            )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, preprocessor: Preprocessor, max_cache_size=10000, num_workers=8
    ):
        super(LazySupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.preprocessor = preprocessor
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._getitem = lru_cache(maxsize=max_cache_size)(self._load_item)

    def __len__(self):
        return len(self.raw_data)

    def _load_item(self, i):
        data = self.raw_data[i]
        if not data or not data.get("conversations"):
            raise ValueError(f"Data at index {i} is invalid.")
        ret = self.preprocessor.preprocess(
            [data["conversations"]], systems=[data.get("system", "")]
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            future = self.executor.submit(self._getitem, i)
            return future.result()
        except Exception as e:
            print(f"Getting item error {e}")
            # return a dummy sample
            return dict(
                input_ids=torch.tensor([]),
                labels=torch.tensor([]),
                attention_mask=torch.tensor([]),
            )


def load_json_data(data_path):
    if data_path.endswith(".json"):
        raw_data = json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, mode="r") as reader:
            raw_data = list(reader)
    return raw_data


def make_supervised_data_module(preprocessor: Preprocessor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = load_json_data(data_args.data_path)
    train_dataset = dataset_cls(train_json, preprocessor=preprocessor)

    if data_args.eval_data_path:
        eval_json = load_json_data(data_args.eval_data_path)
        eval_dataset = dataset_cls(eval_json, preprocessor=preprocessor)
    else:
        eval_dataset = None

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
        # config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
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
    tokenizer.pad_token = tokenizer.unk_token
    model.resize_token_embeddings(
        max(len(tokenizer), config.vocab_size), pad_to_multiple_of=8
    )

    preprocessor = Preprocessor(tokenizer, template_name=model_args.template_name)
    data_module = make_supervised_data_module(
        preprocessor=preprocessor, data_args=data_args
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if training_args.resume_from_checkpoint and list(
        pathlib.Path(training_args.resume_from_checkpoint).glob("checkpoint-*")
    ):
        trainer.train(
            resume_from_checkpoint=get_last_checkpoint(
                training_args.resume_from_checkpoint
            )
        )
    elif list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    if training_args.deepspeed is not None:
        save_deepspeed_model(trainer, tokenizer, output_dir=training_args.output_dir)
    else:
        trainer_save_model_safe(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
