import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json,re

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
LABLE_PROMPT = '请直接告诉我，以下这个回复最有可能对应的指令是什么：'
POINT_PROMPT=("评价指令质量。请根据以下指令的内容，使用【好】或【差】进行评价：\
高质量指令特性：1. 明确性：目标明确，无模糊解释。2. 精确性：详述所需操作，避免模糊词语。3. 完整性：信息完备，执行无需猜测。\
4. 可行性：指令实际可执行，适合接收者能力。5. 简洁性：语言简单明了，避免冗余。6. 逻辑性：按逻辑顺序描述步骤。7. 正确性：基于准确信息，能获得预期结果。8. 可跟踪性：能确认指令执行情况。\
低质量指令特性：1. 模糊：目标或步骤不明确。2. 不完整：信息缺失需额外寻找。3. 不实际：超出接收者能力或资源。4. 冗长复杂：用词复杂，含多余术语。5. 逻辑混乱：步骤顺序不清晰。6. 错误的信息：基于错误或过时信息。\
示例：-【请解释什么是“联合开发”，及其风险与收益。】答：【好】\
-【列出《现场总线技术及Profibus》书目录。】答：【好】\
-【孟 安全职称：副主任医师 科室：内一科 简介：内一科副主任 擅长：呼吸道疾病、小儿哮喘及新生儿疾病诊疗】答：【差】\
-【为以下人物编写简历。】答：【差】"
              )
# POINT_PROMPT="给定以下instruction-output数据对，请为其打分。1分代表低质量的instruction-output数据，而2分代表高质量的数据对。请根据数据对的准确性、完整性和相关性进行评分。输入:"
ANSWER_PROMPT="阅读以下材料："

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
        #self.data = [item for item in self.data if self.is_valid_item(item)]  # 过滤数据
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
        instruction =  example["instruction"] + example['input']
        output = example['output']
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
            r=10,
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
