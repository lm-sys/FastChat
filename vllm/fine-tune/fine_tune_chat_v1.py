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
import ast




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


@dataclass
class DataArguments:
    data_ratios: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing training data files."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training data files."}
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
        data_dir,
        data_ratios,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.tmp = 0
        self.data = []
        
        # List all JSON files in the data_dir
        extraction_ratios = {
            "yayi.jsonl": data_ratios[0], # 去括号
            "OL-CC.jsonl": data_ratios[1],
            "firefly.jsonl": data_ratios[2],
            "WuDao.jsonl":data_ratios[3],
            "COIG_exam_instructions.jsonl":data_ratios[4],
            "COIG_human_value_alignment_instructions_part1.jsonl":data_ratios[5],
            "COIG_human_value_alignment_instructions_part2.jsonl":data_ratios[6],
            "COIG_leetcode_instructions.jsonl":data_ratios[7],
            "kun_55W.jsonl":data_ratios[8],
            "baichuan_55W.jsonl":data_ratios[9],
        }

        # Default extraction ratio for files not in the dictionary
        default_ratio = data_ratios[10]

        # en_ratio = data_ratios[11]  #we have 20W english sft data

        # List all JSON files in the data_dir
        data_files = pathlib.Path(data_dir).glob("*.jsonl")
        for file in data_files:
            with open(file, 'r') as f:
                file_data = [json.loads(line) for line in f]
                
                # Calculate the number of samples to extract and the step size for equal distribution
                ratio = extraction_ratios.get(file.name, default_ratio)  # Use default ratio if file not in ratios
                num_samples = int(len(file_data) * ratio)
                if num_samples == 0:
                    continue
                step_size = len(file_data) // num_samples
                
                # Extract data based on the ratio for the file
                self.data.extend(file_data[::step_size][:num_samples])

        directory = '/cpfs/29cd2992fe666f2a/user/zhangge/xw/Humpback-CH/data/en_sft_data'
        if len(data_ratios)==12:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        self.data.extend(file_data)

        
        # self.data = [item for item in self.data if self.is_valid_item(item)]  # 过滤数据
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))
        print(f"Total number of data samples read: {len(self.data)}")

    

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        # 使用新的数据格式
        instruction = example["instruction"] + example['input']
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
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    data_ratios_str = data_args.data_ratios  # 从命令行参数获取
    data_ratios = ast.literal_eval(data_ratios_str)
    print (data_ratios)
    # 确保解析后的对象是一个列表
    if not isinstance(data_ratios, list):
        raise ValueError("data_ratios should be a list")
    
    dataset = SupervisedDataset(
        data_args.data_dir, data_ratios, tokenizer, training_args.model_max_length
    )
    
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
