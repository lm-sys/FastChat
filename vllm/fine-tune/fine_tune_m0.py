import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json, re

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments

LABLE_PROMPT = '请直接告诉我，以下这个回复最有可能对应的指令是什么：'
POINT_PROMPT = "以下是用户的指示和一个候选答案，请评估该答案是否是AI助手应该如何响应用户指示的良好示例。请使用以下5点评分标准来分配得分：\
    1：意味着答案是不完整的、模糊的、离题的、有争议的或不完全符合用户要求的。例如，内容似乎缺失，编号列表并不是从一开始，开头句子重复了用户的问题。或者响应是从其他人的角度，结合他们的个人经验（例如，摘自博客文章）或看起来像是来自论坛的答案。或者包含宣传文本、导航文本或其他无关的信息。\
    2：意味着答案涵盖了用户的大部分要求。它没有直接回答用户的问题。例如，它只提供了一种高级方法，而不是用户问题的确切解决方案。\
    3：意味着答案是有帮助的，但不是由AI助手写的。它满足了用户的所有基本要求。它是完整的、自给自足的，但缺点是响应不是从AI助手的角度写的，而是从其他人的角度写的。内容看起来像是从博客文章、网页或网页搜索结果中摘录出来的。例如，包含个人经验或意见，提及评论部分，或分享到社交媒体等。\
    4：意味着答案是从AI助手的角度写的，清晰地关注于满足指示。它为用户的问题或指示提供了完整、清晰、全面的答案，没有遗漏或不相关的信息。内容组织有序，自给自足，语气友好。还有少量的改进空间，例如，更简洁和集中。\
    5：意味着这是一个完美的AI助手的答案。其重点明确地是作为一个有帮助的AI助手，响应似乎是专门为了满足用户的问题或指示而写的，没有任何无关的句子。答案提供了高质量的内容，展现了该领域的专家知识，写得非常好，逻辑清晰，易于遵循，引人入胜且富有洞察力。\
    请首先提供你用来得出评分的简要理由，然后在最后一行写上'得分：<评分>'。"
# POINT_PROMPT="给定以下instruction-output数据对，请为其打分。1分代表低质量的instruction-output数据，而2分代表高质量的数据对。请根据数据对的准确性、完整性和相关性进行评分。输入:"
ANSWER_PROMPT = "阅读以下材料："


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


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
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path,
            tokenizer,
            model_max_length,
            user_tokens=[195],
            assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.tmp = 0
        self.data = json.load(open(data_path))
        # self.data = [item for item in self.data if self.is_valid_item(item)]  # 过滤数据
        self.data = [item for item in self.data]
        # self.data = self.data[:int(0.8*len(self.data))]
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        # print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def is_valid_item(self, item):
        # 你可以在这里加入其他的检查逻辑
        input_data = item.get('input', "")
        if all(char in input_data for char in "ABCD"):
            self.tmp += 1
            return False
        return True

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        # 使用新的数据格式
        output = example['output']
        instruction = example["instruction"] + example['input']
        # 对instruction进行编码
        instruction_ids = self.tokenizer.encode(instruction)
        output_ids = self.tokenizer.encode(output)

        # 将instruction添加到input_ids
        input_ids += self.user_tokens + instruction_ids
        labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(instruction_ids)

        # 将output添加到input_ids
        input_ids += self.assistant_tokens + output_ids
        labels += [self.ignore_index] + output_ids

        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # 填充到模型的最大长度
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
