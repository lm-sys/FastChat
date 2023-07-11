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

from collections import defaultdict
import copy
import os
from dataclasses import dataclass, field
import random
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist


from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AddedToken, BitsAndBytesConfig, deepspeed


from fastchat.model.model_adapter import get_conversation_template

default_conversation = get_conversation_template("t5")

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q", "v"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


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



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    num_data: int = -1
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, state_dict: dict):
    """Collects the state dict and dump to disk."""
    
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    other_tokens,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    for new_token in other_tokens:
        num_new_tokens += tokenizer.add_tokens(AddedToken(new_token, normalized=False))

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _form_qa(
    q_list,
    a_list,
    tokenized_conversation,
    tokenized_lens,
    speakers,
    header_len,
    max_length,
    eos_id,
):
    cur_idx = header_len
    conv_len = len(tokenized_conversation)

    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if cur_idx >= conv_len:
            break
        if speaker == "gpt":
            # truncate answer if it is too long
            content_a = None
            if tokenized_len > max_length:
                content_a = tokenized_conversation[cur_idx : cur_idx + max_length]
            else:
                content_a = tokenized_conversation[cur_idx : cur_idx + tokenized_len]
            content_a.append(eos_id)
            a_list.append(content_a)
            content_q = None
            if cur_idx >= max_length:
                content_q = tokenized_conversation[cur_idx - max_length : cur_idx]
            else:
                content_q = tokenized_conversation[:cur_idx]
            content_q.append(eos_id)
            q_list.append(content_q)
            # asser the last token is actually a EOS for an answer
            assert a_list[-1][-1] == eos_id, "Last Token is not EOS!"
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header

    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": default_conversation.roles[0],  # human role
        "gpt": default_conversation.roles[1],  # gpt role
    }

    for i in range(len(source)):
        sentence = source[i]
        sentence_from = sentence["from"].lower()

        # TODO(Dacheng): verify this is a good way to split sentences
        if sentence_from == "human":
            # if this is not the last sentence
            if i != len(source) - 1:
                next_sentence = source[i + 1]
                sentence["value"] = (
                    BEGIN_SIGNAL
                    + roles.get(sentence_from, unknown_role)
                    + ": "
                    + sentence["value"]
                    + END_SIGNAL
                    + BEGIN_SIGNAL
                    + roles.get(next_sentence["from"].lower(), unknown_role)
                    + ": "
                )
            else:
                # if human is the last speaker, it does not contribute to an answer
                pass
        else:
            sentence["value"] = sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]

    return conversation


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    header = f"{default_conversation.system}\n\n"
    for source in sources:
        conversation = _add_speaker_and_signal(header, source, tokenizer)
        conversations.append(conversation)
    # TODO(Dacheng): This is related to whether the dataset has been truncated..
    # Assume we get long conversations, don't pad, don't return tensor
    tokenized_conversations = tokenizer(conversations, max_length=None)["input_ids"]
    q_list = []
    a_list = []
    # count for EOS length
    header_len = _tokenize_fn([header], tokenizer)["input_ids_lens"][0] - 1
    from tqdm import tqdm

    for tokenized_conversation, source in tqdm(zip(tokenized_conversations, sources)):
        tokenized_sentence = _tokenize_fn([s["value"] for s in source], tokenizer)
        tokenized_lens = tokenized_sentence["input_ids_lens"]
        tokenized_lens = [l - 1 for l in tokenized_lens]
        speakers = [sentence["from"] for sentence in source]
        ids = tokenized_sentence["input_ids"]
        _form_qa(
            q_list,
            a_list,
            tokenized_conversation,
            tokenized_lens,
            speakers,
            header_len,
            tokenizer.model_max_length,
            tokenizer.eos_token_id,
        )
    return dict(input_ids=q_list, labels=a_list)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        preprocessed_path,
        num_data,
    ):
        super(SupervisedDataset, self).__init__()

        # save to file
        # Make sure only the first process is processing the dataset
        if dist.get_rank() != 0:
            dist.barrier()
        self.preprocessed_path = preprocessed_path
        if os.path.exists(self.preprocessed_path):
            logging.warning("loading from preprocessed data")
            with open(self.preprocessed_path, "r") as f:
                data_dict = json.load(f)
            # if dist.get_rank() == 0:
            #     dist.barrier()
        else:
            if not os.path.exists("preprocessed_data"):
                os.mkdir("preprocessed_data")
            assert dist.get_rank() == 0, "Only the first process should process"
            logging.warning("Loading data...")
            list_data_dict = json.load(open(data_path, "r"))

            logging.warning("Formatting inputs...")
            sources = []

            sources = [example["conversations"] for example in list_data_dict]

            data_dict = preprocess(sources, tokenizer)
            json_data_dict = json.dumps(data_dict)

            # Remember to close file to avoid concurrent r/w
            with open(self.preprocessed_path, "w") as f:
                f.write(json_data_dict)

            # Release barrier
            dist.barrier()

        if num_data != -1:
            data_dict["input_ids"] = data_dict["input_ids"][:num_data]
            data_dict["labels"] = data_dict["labels"][:num_data]

        # Shuffle data to see more conversations, if only train on partial data
        temp = list(zip(data_dict["input_ids"], data_dict["labels"]))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        data_dict["input_ids"], data_dict["labels"] = list(res1), list(res2)

        # Dacheng: Get rid of short QA pair
        self.input_ids = copy.deepcopy(data_dict["input_ids"])
        self.labels = copy.deepcopy(data_dict["labels"])
        length_arr = defaultdict(int)
        for idx, (input, label) in enumerate(
            zip(data_dict["input_ids"], data_dict["labels"])
        ):
            length_arr[str(len(label) // 100)] += 1
            if len(input) <= 5:
                del_idx = self.input_ids.index(input)
                self.input_ids.pop(del_idx)
                self.labels.pop(del_idx)
            if len(label) <= 5:
                del_idx = self.labels.index(label)
                self.input_ids.pop(del_idx)
                self.labels.pop(del_idx)

        for input, label in zip(self.input_ids, self.labels):
            assert len(input) >= 5
            assert len(label) >= 5

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [
                torch.as_tensor(instance[key], dtype=torch.int64)
                for instance in instances
            ]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        torch.set_printoptions(profile="full")
        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        preprocessed_path=data_args.preprocessed_path,
        num_data=data_args.num_data,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
    )
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # Dacheng: Note we can only use T5Tokenizer, otherwise it will prepend
    # a space before special tokens.
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        other_tokens=["<", "{", "\n", "}", "`", " ", "\\", "^", "\t"],
        tokenizer=tokenizer,
        model=model,
    )

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
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
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, state_dict=state_dict)

if __name__ == "__main__":
    train()