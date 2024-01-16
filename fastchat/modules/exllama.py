from dataclasses import dataclass, field
import sys


@dataclass
class ExllamaConfig:
    max_seq_len: int
    gpu_split: str = None
    cache_8bit: bool = False


class ExllamaModel:
    def __init__(self, exllama_model, exllama_cache):
        self.model = exllama_model
        self.cache = exllama_cache
        self.config = self.model.config


def load_exllama_model(model_path, exllama_config: ExllamaConfig):
    try:
        from exllamav2 import (
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Cache_8bit,
        )
    except ImportError as e:
        print(f"Error: Failed to load Exllamav2. {e}")
        sys.exit(-1)

    exllamav2_config = ExLlamaV2Config()
    exllamav2_config.model_dir = model_path
    exllamav2_config.prepare()
    exllamav2_config.max_seq_len = exllama_config.max_seq_len
    exllamav2_config.cache_8bit = exllama_config.cache_8bit

    exllama_model = ExLlamaV2(exllamav2_config)
    tokenizer = ExLlamaV2Tokenizer(exllamav2_config)

    split = None
    if exllama_config.gpu_split:
        split = [float(alloc) for alloc in exllama_config.gpu_split.split(",")]
    exllama_model.load(split)

    cache_class = ExLlamaV2Cache_8bit if exllamav2_config.cache_8bit else ExLlamaV2Cache
    exllama_cache = cache_class(exllama_model)
    model = ExllamaModel(exllama_model=exllama_model, exllama_cache=exllama_cache)

    return model, tokenizer
