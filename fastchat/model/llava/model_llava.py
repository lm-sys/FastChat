"""
Inference code for Llava.
Adapted from https://huggingface.co/spaces/badayvedat/LLaVA/blob/main/llava/serve/model_worker.py and
https://github.com/haotian-liu/LLaVA/blob/5da97161b9e2c3ae19b1d4a39eeb43148091d728/llava/mm_utils.py
"""

from io import BytesIO
import base64
import json
from threading import Thread

import torch
from transformers import StoppingCriteria, TextIteratorStreamer
from fastchat.model.llava.constants import (
    IMAGE_TOKEN_INDEX,
    LLAVA_IMAGE_TOKEN,
    LLAVA_IM_START_TOKEN,
    LLAVA_IM_END_TOKEN,
)


def load_image_from_base64(image):
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    from PIL import Image

    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


@torch.inference_mode()
def generate_stream_llava(
    model,
    tokenizer,
    image_processor,
    params,
    device,
    context_len=2048,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]
    ori_prompt = prompt
    images = params.get("images", None)
    num_image_tokens = 0
    if (
        images is not None and len(images) > 0
    ):  # NOTE(chris): removed multimodal check because we essentially know we want multimodal on
        if len(images) > 0:
            if len(images) != prompt.count(LLAVA_IMAGE_TOKEN):
                raise ValueError(
                    "Number of images does not match number of <image> tokens in prompt"
                )

            images = [load_image_from_base64(image) for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [
                    image.to(model.device, dtype=torch.float16) for image in images
                ]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = LLAVA_IMAGE_TOKEN
            if getattr(model.config, "mm_use_im_start_end", False):
                replace_token = (
                    LLAVA_IM_START_TOKEN + replace_token + LLAVA_IM_END_TOKEN
                )
            prompt = prompt.replace(LLAVA_IMAGE_TOKEN, replace_token)

            num_image_tokens = (
                prompt.count(replace_token) * model.get_vision_tower().num_patches
            )
        else:
            images = None
        image_args = {"images": images}
    else:
        images = None
        image_args = {}

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    echo = params.get("echo", False)
    do_sample = True if temperature > 0.001 else False

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
    )

    max_new_tokens = min(
        max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
    )

    if max_new_tokens < 1:
        yield {
            "text": ori_prompt
            + "Exceeds max token length. Please start a new conversation, thanks.",
            "error_code": 0,
        }
        return

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args,
        ),
    )
    thread.start()

    if echo:
        generated_text = ori_prompt
    else:
        generated_text = ""

    generated_tokens = 0
    finish_reason = None
    for new_text in streamer:
        generated_text += new_text
        generated_tokens += len(tokenizer.encode(new_text))
        if generated_text.endswith(stop_str):
            finish_reason = "stop"
            break
        elif generated_tokens >= max_new_tokens:
            finish_reason = "length"
            break

        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_ids.shape[-1],
                "completion_tokens": generated_tokens,
                "total_tokens": input_ids.shape[-1] + generated_tokens,
            },
            "finish_reason": None,
        }

    yield {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_ids.shape[-1],
            "completion_tokens": generated_tokens,
            "total_tokens": input_ids.shape[-1] + generated_tokens,
        },
        "finish_reason": finish_reason,
    }
