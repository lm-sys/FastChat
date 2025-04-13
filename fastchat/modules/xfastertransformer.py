from configparser import ConfigParser
from dataclasses import dataclass
import os
import sys


@dataclass
class XftConfig:
    early_stopping: bool = False
    do_sample: bool = False
    is_encoder_decoder: bool = False
    padding: bool = True
    beam_width: int = 1
    eos_token_id: int = -1
    max_seq_len: int = 512
    num_return_sequences: int = 1
    pad_token_id: int = -1
    top_k: int = -1
    repetition_penalty: float = 1.1
    temperature: float = 1.0
    top_p: float = -1.0
    data_type: str = "bf16_fp16"
    model_type: str = ""
    stop_words_ids: str = None


class XftModel:
    def __init__(self, xft_model, xft_config):
        self.model = xft_model
        self.config = xft_config


def get_stop_words_ids_qwen(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def load_default_config(model_path, xft_config):
    file_path = os.path.join(model_path, "config.ini")
    cf = ConfigParser()
    cf.read(file_path, encoding="utf-8")
    sections = cf.sections()
    if len(sections) <= 0:
        return

    xft_config.model_type = sections[0].lower()

    keys = []
    values = []
    for it in cf.items(sections[0]):
        try:
            if it[1] is not None:
                if it[0] == "end_id":
                    xft_config.eos_token_id = (int)(it[1])
                elif it[0] == "do_sample":
                    if it[1].lower() == "true":
                        xft_config.do_sample = True
                    else:
                        xft_config.do_sample = False
                elif it[0] == "pad_id":
                    xft_config.pad_token_id = (int)(it[1])
                elif it[0] == "repetition_penalty":
                    xft_config.repetition_penalty = (float)(it[1])
                elif it[0] == "top_k":
                    xft_config.top_k = (int)(it[1])
                elif it[0] == "top_p":
                    xft_config.top_p = (float)(it[1])
        except ValueError as e:
            print(
                f"Warnning: xFasterTransformer parser config.ini error @{it[0]}={it[1]}. {e}"
            )


def load_xft_model(model_path, xft_config: XftConfig):
    try:
        import xfastertransformer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Failed to load xFasterTransformer. {e}")
        sys.exit(-1)

    if xft_config.data_type is None or xft_config.data_type == "":
        data_type = "bf16_fp16"
    else:
        data_type = xft_config.data_type
    load_default_config(model_path, xft_config)

    revision = "main"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
            revision=revision,
            trust_remote_code=True,
        )
    except Exception:
        print(f"model_path={model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision=revision, trust_remote_code=True
        )

    if "qwen" in xft_config.model_type:
        xft_config.stop_words_ids = get_stop_words_ids_qwen("chatml", tokenizer)

    xft_model = xfastertransformer.AutoModel.from_pretrained(
        model_path, dtype=data_type
    )

    model = XftModel(xft_model=xft_model, xft_config=xft_config)
    if model.model.rank > 0:
        while True:
            model.model.generate()
    return model, tokenizer
