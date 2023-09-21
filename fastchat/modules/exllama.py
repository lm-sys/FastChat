from dataclasses import dataclass, field
import sys


@dataclass
class ExllamaConfig:
    max_seq_len: int
    gpu_split: str = None


def load_exllama_model(model_path, exllama_config: ExllamaConfig):
    try:
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
    except ImportError as e:
        print(f"Error: Failed to load Exllamav2. {e}")
        sys.exit(-1)

    exllamav2_config = ExLlamaV2Config()
    exllamav2_config.model_dir = model_path
    exllamav2_config.prepare()
    exllamav2_config.max_seq_len = exllama_config.max_seq_len

    model = ExLlamaV2(exllamav2_config)
    tokenizer = ExLlamaV2Tokenizer(exllamav2_config)
    split = None
    if exllama_config.gpu_split:
        split = [float(alloc) for alloc in exllama_config.gpu_split.split(",")]
    model.load(split)

    return model, tokenizer


def init_exllama_cache(model):
    try:
        from exllamav2 import ExLlamaV2Cache
    except ImportError as e:
        print(f"Error: Failed to load Exllamav2. {e}")
        sys.exit(-1)
    return ExLlamaV2Cache(model)
