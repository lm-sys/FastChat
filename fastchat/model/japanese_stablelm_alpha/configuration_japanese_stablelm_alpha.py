# coding=utf-8
# Copyright 2023 Stability and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" JapaneseStableLMAlpha model configuration"""

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

STABLE_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class JapaneseStableLMAlphaConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the JapaneseStableLMAlphaModel. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`JapaneseStableLMAlphaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the decoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        rotary_pct (`float`, *optional*, defaults to 0.25):
            Percentage of hidden dimensions to allocate to rotary embeddings.
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            Base for computing rotary embeddings frequency.
        rotary_scale_base (`int`, *optional*, defaults to 512)
            Base `scale` for computing XPos rotary embeddings scale.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model
            [`StableLMForTokenClassification`]. The dropout ratio for the hidden layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions
            (not used by all models). Only relevant if `config.is_decoder=True`.
        use_parallel_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a "parallel" formulation in each Transformer layer,
            which can provide a slight training speedup at large scales.
        Example:

    ```python
    >>> from transformers import JapaneseStableLMAlphaConfig, JapaneseStableLMAlphaModel

    >>> # Initializing a JapaneseStableLMAlpha style configuration
    >>> configuration = JapaneseStableLMAlphaConfig()

    >>> # Initializing a model (with random weights) from the style configuration
    >>> model = JapaneseStableLMAlphaModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""
    def __init__(
        self,
        vocab_size=65536,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        rotary_scale_base=512,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=3,
        eos_token_id=3,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        use_bias_in_mlp=True,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rotary_scale_base = rotary_scale_base
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.use_bias_in_mlp = use_bias_in_mlp
