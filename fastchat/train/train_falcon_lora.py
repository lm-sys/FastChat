# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

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

from dataclasses import dataclass, field
import logging
import pathlib
import typing

import torch

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
import transformers
from transformers import Trainer

from train_falcon import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module
)

# TODO add support of flash attention in falcon
# from fastchat.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

#TODO: figure out if only apply lora to q and v matrix is enough or not
@dataclass
class NIOArguments:
    nio_lock_file_abspath: str = "/root/lock_file.json"
    nio_task_id: str = "none"
    nio_train_ratio: float = 0.98

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["query_key_value"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, NIOArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        nio_arguments,
    ) = parser.parse_args_into_dataclasses()

    try:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            with open(nio_arguments.nio_lock_file_abspath,'w') as f:
                f.write(f'task_id:{nio_arguments.nio_task_id} status:running')

        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=training_args.cache_dir,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if training_args.deepspeed is not None and training_args.local_rank == 0:
            model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            logging.warning(
                "gradient checkpointing with lora makes requires_grad "
                "incorrect and needs a monkey patch in Trainer or the "
                "wrapped model's forward. ref: "
                "https://github.com/lm-sys/FastChat/pull/138#issuecomment-1509172198"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False)
        tokenizer.pad_token_id = 9

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, train_ratio=nio_arguments.nio_train_ratio)
        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )

        model.config.use_cache = False

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        # check if zero3 mode enabled
        if trainer.hf_deepspeed_config_orig.is_zero3():
            # use deepspeed engine internal function to gather state dict
            # state_dict_zero3 contains whole parameters of base and lora adapters
            # we will not extract lora parameters since peft save_pretrained will do that
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
            state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
            if training_args.local_rank == 0:
                state_dict = state_dict_zero3
        else:
            # in other mode we use original code from fastchat team, to make sure our change is minimum
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), lora_args.lora_bias
            )

        if training_args.local_rank == 0:
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    
    except Exception as e:
        # let rank 0 in distributed mode or rank -1 in single machine mode modify lock file
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            with open(nio_arguments.nio_lock_file_abspath,'w') as f:
                f.write(f'task_id:{nio_arguments.nio_task_id} status:error')
        raise e


if __name__ == "__main__":
    train()
