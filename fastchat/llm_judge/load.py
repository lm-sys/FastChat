# Copyright (c) 2023, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""Tokenization classes for xgen-15B (gpt4 tokenizer)"""

from typing import List, Optional

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
from transformers import AutoModelForCausalLM

try:
    import tiktoken
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("XGen requires the installation of tiktoken. Please install it via `pip install tiktoken`.") from e


logger = logging.get_logger(__name__)


def tiktoken_tokenizer(base="gpt-4", pad_token=None, add_special=True):
    if not add_special:
        return tiktoken.get_encoding(base)

    def include_fim_tokens():
        fim_tokens = [
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<empty_output>",
            "<commit_before>",
            "<commit_msg>",
            "<commit_after>",
            "<reponame>"
        ]
        return fim_tokens

    fim_tokens = include_fim_tokens()

    # tokenizer = tiktoken.get_encoding(base)
    tokenizer = tiktoken.encoding_for_model(base)
    idx = tokenizer.n_vocab

    bpe_ranks = tokenizer._mergeable_ranks

    special_tokens = dict()

    for sp in fim_tokens:
        special_tokens[sp] = idx
        idx += 1

    # if pad_token and pad_token not in tokenizer._special_tokens and pad_token not in special_tokens:
    #     special_tokens[pad_token] = idx
    #     idx += 1
    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name=base.replace("base", "im"),
        pat_str=tokenizer._pat_str,
        mergeable_ranks=bpe_ranks,
        special_tokens={
            **tokenizer._special_tokens,
            **special_tokens
        }
    )
    return enc


class Xgen15BTokenizer(PreTrainedTokenizer):
    """
    Construct a Xgen tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            pad_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            add_eos_token=False,
            add_special_tokens=True,
            max_seq_length=2048*8,
            **kwargs,
    ):
        pad_token_added = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token_added = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        self.add_eos_token = add_eos_token
        self.encoder = tiktoken_tokenizer(base="gpt-4", pad_token=pad_token, add_special=add_special_tokens)
        super().__init__(
            pad_token=pad_token_added,
            eos_token=eos_token_added,
            add_eos_token=add_eos_token,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        self.model_max_length = max_seq_length
        self.pad_token_id = self.encoder._special_tokens['<|endoftext|>']
        self.eos_token_id = self.encoder._special_tokens['<|endoftext|>']

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.encoder.n_vocab

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {}
        for i in range(self.vocab_size):
            try:
                key = self.encoder.decode_single_token_bytes(i)
                vocab[key] = i
            except Exception:
                # TODO - handle this better
                # This is a hacky way to 
                # enable support for latest 
                # version of transformers (4.38)
                # instead of v4.33...
                # keep an eye out for issues here
                continue        
        return vocab

    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        return self.encoder.encode(text, allowed_special="all")

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if isinstance(token, str):
            return self.encoder.encode_single_token(token)
        else:
            return token

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.encoder.decode_single_token_bytes(index).decode("utf-8")

    def _decode(self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs):
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in self.all_special_ids]
        return self.encoder.decode(token_ids)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + eos_token_id
        return ([0] * len(token_ids_0)) + eos_token_id + ([0] * len(token_ids_1)) + eos_token_id

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        if token_ids_1 is None, only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(token_ids_1 + eos_token_id)

        return output

    # has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()
