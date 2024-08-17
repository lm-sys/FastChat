import datetime
import hashlib
import os
import json
import time
import base64
import requests
from typing import Tuple, Dict, List, Union

from fastchat.constants import LOGDIR
from fastchat.serve.vision.image import Image
from fastchat.utils import load_image, upload_image_file_to_gcs


class BaseContentModerator:
    def __init__(self):
        raise NotImplementedError

    def image_moderation_filter(self, image: Image) -> Tuple[bool, bool]:
        raise NotImplementedError

    def text_moderation_filter(self, text: str) -> bool:
        raise NotImplementedError

    def write_to_json(self):
        raise NotImplementedError


class AzureAndOpenAIContentModerator(BaseContentModerator):
    def __init__(self, use_remote_storage: bool = False):
        """
        This class is used to moderate content using Azure and OpenAI.

        text_and_openai_moderation_responses is a list of dictionaries that holds content and OpenAI moderation responses
        image_and_azure_moderation_responses is a list of dictionaries that holds image and Azure moderation responses
        """
        self.conv_to_moderation_responses: Dict[
            str, Dict[str, Union[str, Dict[str, float]]]
        ] = {}
        self.use_remote_storage = use_remote_storage

    def write_to_json(self, ip):
        t = datetime.datetime.now()
        conv_log_filename = f"toxic-{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
        with open(conv_log_filename, "a") as f:
            if self.conv_to_moderation_responses:
                res = {
                    "tstamp": round(time.time(), 4),
                    "ip": ip,
                }
                res.update(self.conv_to_moderation_responses)
                f.write(json.dumps(res) + "\n")

        self.conv_to_moderation_responses = {}

    def _image_moderation_request(
        self, image_bytes: bytes, endpoint: str, api_key: str
    ) -> dict:
        headers = {"Content-Type": "image/jpeg", "Ocp-Apim-Subscription-Key": api_key}

        MAX_RETRIES = 3
        for _ in range(MAX_RETRIES):
            response = requests.post(endpoint, headers=headers, data=image_bytes).json()
            try:
                if response["Status"]["Code"] == 3000:
                    break
            except:
                time.sleep(0.5)
        return response

    def _image_moderation_provider(self, image_bytes: bytes, api_type: str) -> bool:
        if api_type == "nsfw":
            endpoint = os.environ["AZURE_IMG_MODERATION_ENDPOINT"]
            api_key = os.environ["AZURE_IMG_MODERATION_API_KEY"]
            response = self._image_moderation_request(image_bytes, endpoint, api_key)
            flagged = response["IsImageAdultClassified"]
        elif api_type == "csam":
            endpoint = (
                "https://api.microsoftmoderator.com/photodna/v1.0/Match?enhance=false"
            )
            api_key = os.environ["PHOTODNA_API_KEY"]
            response = self._image_moderation_request(image_bytes, endpoint, api_key)
            flagged = response["IsMatch"]

        if flagged:
            image_md5_hash = hashlib.md5(image_bytes).hexdigest()
            self.conv_to_moderation_responses[f"{api_type}_api_moderation"] = {
                "image_hash": image_md5_hash,
                "response": response,
            }

        return flagged

    def image_moderation_filter(self, image: Image) -> Tuple[bool, bool]:
        print(f"moderating image")

        image_bytes = base64.b64decode(image.base64_str)

        nsfw_flagged = self._image_moderation_provider(image_bytes, "nsfw")
        csam_flagged = False

        if nsfw_flagged:
            csam_flagged = self._image_moderation_provider(image_bytes, "csam")

        if nsfw_flagged or csam_flagged:
            image_md5_hash = hashlib.md5(image_bytes).hexdigest()
            directory = "serve_images" if not csam_flagged else "csam_images"
            filename = os.path.join(
                directory,
                f"{image_md5_hash}.{image.filetype}",
            )
            loaded_image = load_image(image.base64_str)
            if self.use_remote_storage and not csam_flagged:
                upload_image_file_to_gcs(loaded_image, filename)
            else:
                filename = os.path.join(LOGDIR, filename)
                if not os.path.isfile(filename):
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    loaded_image.save(filename)

        return nsfw_flagged, csam_flagged

    def _openai_moderation_filter(
        self, text: str, custom_thresholds: dict = None
    ) -> bool:
        """
        Check whether the text violates OpenAI moderation API.
        """
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # default to true to be conservative
        flagged = True
        MAX_RETRY = 3
        for _ in range(MAX_RETRY):
            try:
                res = client.moderations.create(input=text)
                flagged = res.results[0].flagged
                if custom_thresholds is not None:
                    for category, threshold in custom_thresholds.items():
                        if (
                            getattr(res.results[0].category_scores, category)
                            > threshold
                        ):
                            flagged = True
                    self.conv_to_moderation_responses["text_moderation"] = {
                        "content": text,
                        "response": dict(res.results[0].category_scores),
                    }
                break
            except (openai.OpenAIError, KeyError, IndexError) as e:
                print(f"MODERATION ERROR: {e}\nInput: {text}")

        return flagged

    def text_moderation_filter(
        self, text: str, model_list: List[str], do_moderation: bool = False
    ) -> bool:
        # Apply moderation for below models
        MODEL_KEYWORDS = [
            "claude",
            "gpt",
            "bard",
            "mistral-large",
            "command-r",
            "dbrx",
            "gemini",
            "reka",
        ]

        custom_thresholds = {"sexual": 0.3}
        # set a stricter threshold for claude
        for model in model_list:
            if "claude" in model:
                custom_thresholds = {"sexual": 0.2}

        for keyword in MODEL_KEYWORDS:
            for model in model_list:
                if keyword in model:
                    do_moderation = True
                    break

        if do_moderation:
            return self._openai_moderation_filter(text, custom_thresholds)
        return False
