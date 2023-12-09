"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

from abc import ABC
import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict, Union, Tuple

import transformers
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()


@dataclasses.dataclass
class PromptGenerator(ABC):
    # The system prompt
    system_prompt: str = ""
    # The names of two roles
    roles: List[str] = ("USER", "ASSISTANT")
    sep: str = "\n"
    sep2: str = None
    role_sep: str = ": "
    gen_role_sep: str = None
    sys_sep: str = None
    im_start: str = ""

    def __post_init__(
        self,
        **kwargs,
    ):
        self.sep2 = self.sep if self.sep2 is None else self.sep2
        self.gen_role_sep = (
            self.role_sep if self.gen_role_sep is None else self.gen_role_sep
        )
        self.sys_sep = self.sep if self.sys_sep is None else self.sys_sep
        self.im_start = self.im_start or ""

    def round_prompt(self, role, message, last=False, **kwargs) -> str:
        turn_ret = ""
        if message:
            turn_ret = f"{role}{self.role_sep}{message}"
            turn_ret = (
                f"{self.im_start}{turn_ret}{self.sep}"
                if role == self.roles[0]
                else f"{turn_ret}{self.sep2}"
            )
        elif role == self.roles[1] and last:
            turn_ret = f"{role}{self.gen_role_sep}"

        return turn_ret

    def get_sys_ret(
        self,
    ):
        return f"{self.system_prompt}{self.sys_sep}" if self.system_prompt else ""

    def get_prompt(self, messages: List[List[str]] = (), **kwargs) -> str:
        msg_len = len(messages)
        ret = self.get_sys_ret()
        for i, (role, message) in enumerate(messages):
            ret += self.round_prompt(role, message, last=(i == msg_len - 1), **kwargs)
        return ret

    def set_tokenizer(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

    def init_basic_info(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        **kwargs,
    ):
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        role_sep = self.role_sep.strip(" ")
        self.human_ids = tokenizer.encode(
            f"{self.roles[0]}{role_sep}", add_special_tokens=False
        )
        self.human_ids_len = len(self.human_ids)
        self.gpt_ids = tokenizer.encode(
            f"{self.roles[1]}{role_sep}", add_special_tokens=False
        )
        self.gpt_ids_len = len(self.gpt_ids)

        # setting bos and eos token_ids
        alter_bos_id = (
            [tokenizer.bos_token_id] if kwargs and kwargs.get("use_alter_bos") else []
        )
        self.bos_token_ids = (
            tokenizer.encode(self.im_start, add_special_tokens=False)
            if self.im_start
            else alter_bos_id
        )
        self.bos_ids_len = len(self.bos_token_ids)

        # TODO: make eos alone with sep2
        alter_eos_id = (
            [tokenizer.eos_token_id] if kwargs and kwargs.get("use_alter_eos") else []
        )
        self.eos_token_ids = (
            tokenizer.encode(self.sep2.strip(" "), add_special_tokens=False)
            if self.sep2
            else alter_eos_id
        )
        self.eos_ids_len = len(self.eos_token_ids)

        self.msg_sep_ids = (
            tokenizer.encode(self.sep.strip(" "), add_special_tokens=False)
            if self.sep
            else []
        )
        self.msg_sep_ids_len = len(self.msg_sep_ids)

        self.sys_ids = (
            tokenizer.encode(
                f"{self.system_prompt}{self.sys_sep.strip(' ')}",
                add_special_tokens=False,
            )
            if self.system_prompt
            else []
        )
        self.sys_ids_len = len(self.sys_ids)
        self.basic_inited = True

    def round_tokens(
        self,
        role: str,
        message: str,
        tokenizer: transformers.PreTrainedTokenizer = None,
        last=False,
        **kwargs,
    ):
        if not hasattr(self, "basic_inited"):
            self.init_basic_info(tokenizer, **kwargs)
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        turn_ids = []
        turn_labels = []
        if message:
            value_ids = tokenizer.encode(message, add_special_tokens=False)
            if role == self.roles[0]:
                turn_ids = (
                    self.bos_token_ids + self.human_ids + value_ids + self.msg_sep_ids
                )
                turn_labels = [IGNORE_TOKEN_ID] * (
                    self.bos_ids_len
                    + self.human_ids_len
                    + len(value_ids)
                    + self.msg_sep_ids_len
                )
            else:
                turn_ids = self.gpt_ids + value_ids + self.eos_token_ids
                turn_labels = (
                    [IGNORE_TOKEN_ID] * self.gpt_ids_len
                    + value_ids
                    + self.eos_token_ids
                )
        elif role == self.roles[1] and last:
            gen_role_ids = tokenizer.encode(
                f"{self.roles[1]}{self.gen_role_sep.strip(' ')}",
                add_special_tokens=False,
            )
            turn_ids = gen_role_ids
            turn_labels = [IGNORE_TOKEN_ID] * len(gen_role_ids)

        return turn_ids, turn_labels

    def get_prompt_token_ids(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        messages: List[List[str]] = (),
        **kwargs,
    ) -> Tuple[List[int]]:
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        msg_len = len(messages)
        self.init_basic_info(tokenizer, **kwargs)

        input_ids = []
        labels = []

        input_ids += self.sys_ids
        labels += [IGNORE_TOKEN_ID] * self.sys_ids_len
        for i, (role, message) in enumerate(messages):
            turn_ids, turn_labels = self.round_tokens(
                role, message, tokenizer, last=(i == msg_len - 1), **kwargs
            )
            input_ids += turn_ids
            labels += turn_labels

        return input_ids, labels


# TODO: add train the query loss, mask the query, which the chat seq change from [sys, user, >gpt, user, >gpt, ...] to [sys (+ greeting), >user, gpt, >user, gpt, >user, ...]


@dataclasses.dataclass
class AddColonSpaceSinglePromptGenerator(PromptGenerator):
    pass


@dataclasses.dataclass
class AddColonSinglePromptGenerator(PromptGenerator):
    gen_role_sep: str = ":"


@dataclasses.dataclass
class AddColonTwoPromptGenerator(PromptGenerator):
    gen_role_sep: str = ":"


@dataclasses.dataclass
class AddNewLineSinglePromptGenerator(PromptGenerator):
    role_sep: str = "\n"


@dataclasses.dataclass
class NoColonSinglePromptGenerator(PromptGenerator):
    role_sep: str = ""
    sys_sep: str = ""


@dataclasses.dataclass
class NoColonTwoPromptGenerator(PromptGenerator):
    role_sep: str = ""
    sys_sep: str = ""


@dataclasses.dataclass
class RWKVPromptGenerator(PromptGenerator):
    gen_role_sep: str = ":"
    sys_sep: str = ""


@dataclasses.dataclass
class LLAMA2PromptGenerator(PromptGenerator):
    role_sep: str = " "
    sys_sep: str = "\n\n"
    im_start: str = "<s>"

    def round_prompt(self, role, message, last=False, **kwargs) -> str:
        turn_ret = ""
        is_first = kwargs.get("is_first", False) if kwargs else False
        sys_ret = kwargs.get("sys_ret", "") if kwargs else ""
        if message:
            if is_first:
                turn_ret = f"{self.roles[0]}{self.role_sep}{sys_ret}"
                if role == self.roles[0]:
                    turn_ret = f"{self.im_start}{turn_ret}{message}{self.sep}"
                else:
                    turn_ret = f"{turn_ret}{role}{self.role_sep}{message}{self.sep2}"
            else:
                turn_ret = f"{role}{self.role_sep}{message}"
                turn_ret = (
                    f"{self.im_start}{turn_ret}{self.sep}"
                    if role == self.roles[0]
                    else f"{turn_ret}{self.sep2}"
                )
        elif role == self.roles[1] and last:
            turn_ret = f"{role}{self.role_sep}"

        return turn_ret

    def get_prompt(self, messages: List[List[str]] = (), **kwargs) -> str:
        msg_len = len(messages)
        sys_ret = self.get_sys_ret()
        return "".join(
            self.round_prompt(
                role,
                message,
                last=(i == msg_len - 1),
                is_first=(i == 0),
                sys_ret=sys_ret,
                **kwargs,
            )
            for i, (role, message) in enumerate(messages)
        )

    def round_tokens(
        self,
        role: str,
        message: str,
        tokenizer: transformers.PreTrainedTokenizer = None,
        last=False,
        **kwargs,
    ):
        if not hasattr(self, "basic_inited"):
            self.init_basic_info(tokenizer, **kwargs)
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        turn_ids = []
        turn_labels = []

        is_first = kwargs.get("is_first", False) if kwargs else False
        if message:
            value_ids = tokenizer.encode(message, add_special_tokens=False)
            if is_first:
                turn_ids = self.human_ids + self.sys_ids
                turn_labels = [IGNORE_TOKEN_ID] * (
                    self.human_ids_len + self.sys_ids_len
                )
                if role == self.roles[0]:
                    turn_ids += value_ids + self.msg_sep_ids
                    turn_labels += [IGNORE_TOKEN_ID] * (
                        len(value_ids) + self.msg_sep_ids_len
                    )
                else:
                    turn_ids += self.gpt_ids + value_ids + self.eos_token_ids
                    turn_labels += (
                        [IGNORE_TOKEN_ID] * self.gpt_ids_len
                        + value_ids
                        + self.eos_token_ids
                    )
            elif role == self.roles[0]:
                turn_ids = (
                    self.bos_token_ids + self.human_ids + value_ids + self.msg_sep_ids
                )
                turn_labels = [IGNORE_TOKEN_ID] * (
                    self.bos_ids_len
                    + self.human_ids_len
                    + len(value_ids)
                    + self.msg_sep_ids_len
                )
            else:
                turn_ids = self.gpt_ids + value_ids + self.eos_token_ids
                turn_labels = (
                    [IGNORE_TOKEN_ID] * self.gpt_ids_len
                    + value_ids
                    + self.eos_token_ids
                )
        elif role == self.roles[1] and last:
            gen_role_ids = tokenizer.encode(
                f"{self.roles[1]}{self.gen_role_sep.strip(' ')}",
                add_special_tokens=False,
            )
            turn_ids = gen_role_ids
            turn_labels = [IGNORE_TOKEN_ID] * len(gen_role_ids)
        return turn_ids, turn_labels

    def get_prompt_token_ids(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        messages: List[List[str]] = (),
        **kwargs,
    ) -> Tuple[List[int]]:
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        msg_len = len(messages)
        self.init_basic_info(tokenizer, **kwargs)

        input_ids = []
        labels = []

        for i, (role, message) in enumerate(messages):
            turn_ids, turn_labels = self.round_tokens(
                role,
                message,
                tokenizer,
                last=(i == msg_len - 1),
                is_first=(i == 0),
                **kwargs,
            )
            input_ids += turn_ids
            labels += turn_labels

        return input_ids, labels


# source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
# source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
@dataclasses.dataclass
class ChatGLMPromptGenerator(PromptGenerator):
    role_sep: str = "："

    def round_prompt(self, role, message, last=False, **kwargs) -> str:
        turn_ret = ""
        rd_start = (
            kwargs.get("rd_start", f"[Round 0]{self.sep}")
            if kwargs
            else f"[Round 0]{self.sep}"
        )
        if message:
            turn_ret = f"{role}{self.role_sep}{message}"
            turn_ret += (
                f"{rd_start}{turn_ret}{self.sep}"
                if role == self.roles[0]
                else f"{turn_ret}{self.sep2}"
            )
        elif role == self.roles[1] and last:
            turn_ret = f"{role}{self.gen_role_sep}"

        return turn_ret

    def get_prompt(self, messages: List[List[str]] = (), **kwargs) -> str:
        msg_len = len(messages)
        ret = self.get_sys_ret()
        round_add_n = 1 if kwargs and kwargs.get("name") == "chatglm2" else 0
        for i, (role, message) in enumerate(messages):
            rd_start = f"[Round {i//2 + round_add_n}]{self.sep}"
            ret += self.round_prompt(
                role,
                message,
                last=(i == msg_len - 1),
                rd_start=rd_start,
                **kwargs,
            )
        return ret

    def round_tokens(
        self,
        role: str,
        message: str,
        tokenizer: transformers.PreTrainedTokenizer = None,
        last=False,
        **kwargs,
    ):
        if not hasattr(self, "basic_inited"):
            self.init_basic_info(tokenizer, **kwargs)
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        turn_ids = []
        turn_labels = []

        default_rd_ids = tokenizer.encode(
            f"[Round 0]{self.sep}",
            add_special_tokens=False,
        )
        rd_start_ids = (
            kwargs.get("rd_start_ids", default_rd_ids) if kwargs else default_rd_ids
        )
        if message:
            value_ids = tokenizer.encode(message, add_special_tokens=False)
            if role == self.roles[0]:
                turn_ids = rd_start_ids + self.human_ids + value_ids + self.msg_sep_ids
                turn_labels = [IGNORE_TOKEN_ID] * (
                    len(rd_start_ids)
                    + self.human_ids_len
                    + len(value_ids)
                    + self.msg_sep_ids_len
                )
            else:
                turn_ids = self.gpt_ids + value_ids + self.eos_token_ids
                turn_labels = (
                    [IGNORE_TOKEN_ID] * self.gpt_ids_len
                    + value_ids
                    + self.eos_token_ids
                )
        elif role == self.roles[1] and last:
            gen_role_ids = tokenizer.encode(
                f"{self.roles[1]}{self.gen_role_sep.strip(' ')}",
                add_special_tokens=False,
            )
            turn_ids = gen_role_ids
            turn_labels = [IGNORE_TOKEN_ID] * len(gen_role_ids)
        return turn_ids, turn_labels

    def get_prompt_token_ids(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        messages: List[List[str]] = (),
        **kwargs,
    ) -> Tuple[List[int]]:
        if not tokenizer:
            tokenizer = self.tokenizer
        if not tokenizer:
            raise ValueError("Tokenizer not set.")
        msg_len = len(messages)
        self.init_basic_info(tokenizer, **kwargs)

        input_ids = []
        labels = []

        input_ids += self.sys_ids
        labels += [IGNORE_TOKEN_ID] * self.sys_ids_len

        round_add_n = 1 if kwargs and kwargs.get("name") == "chatglm2" else 0
        for i, (role, message) in enumerate(messages):
            rd_start_ids = tokenizer.encode(
                f"[Round {i//2 + round_add_n}]{self.sep.strip(' ')}",
                add_special_tokens=False,
            )
            turn_ids, turn_labels = self.round_tokens(
                role,
                message,
                tokenizer,
                last=(i == msg_len - 1),
                rd_start_ids=rd_start_ids,
                **kwargs,
            )
            input_ids += turn_ids
            labels += turn_labels

        return input_ids, labels


@dataclasses.dataclass
class ChatMLPromptGenerator(PromptGenerator):
    role_sep: str = "\n"

    def __post_init__(
        self,
    ):
        super().__post_init__()
        self.sep = f"{self.sep}\n"


# source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
@dataclasses.dataclass
class ChatInternPromptGenerator(PromptGenerator):
    role_sep: str = ":"
    sys_sep: str = ""
    im_start: str = "<s>"

    def __post_init__(
        self,
    ):
        super().__post_init__()
        self.sep = f"{self.sep}\n"
        self.sep2 = f"{self.sep2}\n"


@dataclasses.dataclass
class DollyPromptGenerator(PromptGenerator):
    role_sep: str = ":\n"
    sys_sep: str = ""

    def __post_init__(
        self,
    ):
        super().__post_init__()
        self.sep2 = f"{self.sep2}\n\n"


@dataclasses.dataclass
class PhoenixPromptGenerator(PromptGenerator):
    role_sep: str = ": <s>"
    sys_sep: str = ""


@dataclasses.dataclass
class RobinPromptGenerator(PromptGenerator):
    role_sep: str = ":\n"


@dataclasses.dataclass
class FalconChatPromptGenerator(PromptGenerator):
    gen_role_sep: str = ":"


class PromptGeneratorManager:
    def __init__(self):
        self.generators = {
            SeparatorStyle.ADD_COLON_SINGLE: AddColonSinglePromptGenerator,
            SeparatorStyle.ADD_COLON_SPACE_SINGLE: AddColonSpaceSinglePromptGenerator,
            SeparatorStyle.ADD_COLON_TWO: AddColonTwoPromptGenerator,
            SeparatorStyle.ADD_NEW_LINE_SINGLE: AddNewLineSinglePromptGenerator,
            SeparatorStyle.NO_COLON_SINGLE: NoColonSinglePromptGenerator,
            SeparatorStyle.NO_COLON_TWO: NoColonTwoPromptGenerator,
            SeparatorStyle.RWKV: RWKVPromptGenerator,
            SeparatorStyle.LLAMA2: LLAMA2PromptGenerator,
            SeparatorStyle.CHATGLM: ChatGLMPromptGenerator,
            SeparatorStyle.CHATML: ChatMLPromptGenerator,
            SeparatorStyle.CHATINTERN: ChatInternPromptGenerator,
            SeparatorStyle.DOLLY: DollyPromptGenerator,
            SeparatorStyle.PHOENIX: PhoenixPromptGenerator,
            SeparatorStyle.ROBIN: RobinPromptGenerator,
            SeparatorStyle.FALCON_CHAT: FalconChatPromptGenerator,
        }

    def get_generator(self, style: SeparatorStyle):
        return self.generators.get(style)
