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
        self.conv_moderation_responses: List[
            Dict[str, Dict[str, Union[str, Dict[str, float]]]]
        ] = []
        self.text_flagged = False
        self.csam_flagged = False
        self.nsfw_flagged = False

    def _image_moderation_filter(self, image: Image) -> Tuple[bool, bool]:
        """Function that detects whether image violates moderation policies.

        Returns:
            Tuple[bool, bool]: A tuple of two boolean values indicating whether the image was flagged for nsfw and csam respectively.
        """
        raise NotImplementedError

    def _text_moderation_filter(self, text: str) -> bool:
        """Function that detects whether text violates moderation policies.

        Returns:
            bool: A boolean value indicating whether the text was flagged.
        """
        raise NotImplementedError

    def reset_moderation_flags(self):
        self.text_flagged = False
        self.csam_flagged = False
        self.nsfw_flagged = False

    def image_and_text_moderation_filter(
        self, image: Image, text: str
    ) -> Dict[str, Dict[str, Union[str, Dict[str, float]]]]:
        """Function that detects whether image and text violate moderation policies.

        Returns:
            Dict[str, Dict[str, Union[str, Dict[str, float]]]]: A dictionary that maps the type of moderation (text, nsfw, csam) to a dictionary that contains the moderation response.
        """
        raise NotImplementedError

    def append_moderation_response(
        self, moderation_response: Dict[str, Dict[str, Union[str, Dict[str, float]]]]
    ):
        """Function that appends the moderation response to the list of moderation responses."""
        if (
            len(self.conv_moderation_responses) == 0
            or self.conv_moderation_responses[-1] is not None
        ):
            self.conv_moderation_responses.append(moderation_response)
        else:
            self.update_last_moderation_response(moderation_response)

    def update_last_moderation_response(
        self, moderation_response: Dict[str, Dict[str, Union[str, Dict[str, float]]]]
    ):
        """Function that updates the last moderation response."""
        self.conv_moderation_responses[-1] = moderation_response


class AzureAndOpenAIContentModerator(BaseContentModerator):
    _NON_TOXIC_IMAGE_MODERATION_MAP = {
        "nsfw_moderation": {"flagged": False},
        "csam_moderation": {"flagged": False},
    }

    def __init__(self, use_remote_storage: bool = False):
        """This class is used to moderate content using Azure and OpenAI.

        conv_to_moderation_responses: A dictionary that is a map from the type of moderation
        (text, nsfw, csam) moderation to the moderation response returned from the request sent
        to the moderation API.
        """
        super().__init__()

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

        image_md5_hash = hashlib.md5(image_bytes).hexdigest()
        moderation_response_map = {
            "image_hash": image_md5_hash,
            "response": response,
            "flagged": False,
        }
        if flagged:
            moderation_response_map["flagged"] = True

        return moderation_response_map

    def image_moderation_filter(self, image: Image):
        print(f"moderating image")

        image_bytes = base64.b64decode(image.base64_str)

        nsfw_flagged_map = self._image_moderation_provider(image_bytes, "nsfw")

        if nsfw_flagged_map["flagged"]:
            csam_flagged_map = self._image_moderation_provider(image_bytes, "csam")
        else:
            csam_flagged_map = {"flagged": False}

        self.nsfw_flagged = nsfw_flagged_map["flagged"]
        self.csam_flagged = csam_flagged_map["flagged"]

        # We save only the boolean value instead of the entire response dictionary
        # to save space. nsfw_flagged_map and csam_flagged_map will contain the whole dictionary
        return {
            "nsfw_moderation": {"flagged": self.nsfw_flagged},
            "csam_moderation": {"flagged": self.csam_flagged},
        }

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
        moderation_response_map = {"content": text, "response": None, "flagged": False}
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
                    moderation_response_map = {
                        "response": dict(res.results[0].category_scores),
                        "flagged": flagged,
                    }
                break
            except (openai.OpenAIError, KeyError, IndexError) as e:
                print(f"MODERATION ERROR: {e}\nInput: {text}")

        return moderation_response_map

    def text_moderation_filter(
        self, text: str, model_list: List[str], do_moderation: bool = False
    ):
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

        moderation_response_map = {"flagged": False}
        if do_moderation:
            # We save the entire response dictionary here
            moderation_response_map = self._openai_moderation_filter(
                text, custom_thresholds
            )
            self.text_flagged = moderation_response_map["flagged"]
        else:
            self.text_flagged = False

        # We only save whether the text was flagged or not instead of the entire response dictionary
        # to save space. moderation_response_map will contain the whole dictionary
        return {"text_moderation": {"flagged": self.text_flagged}}

    def image_and_text_moderation_filter(
        self, image: Image, text: str, model_list: List[str], do_moderation=True
    ) -> Dict[str, bool]:
        """Function that detects whether image and text violate moderation policies using the Azure and OpenAI moderation APIs.

        Returns:
            Dict[str, Dict[str, Union[str, Dict[str, float]]]]: A dictionary that maps the type of moderation (text, nsfw, csam) to a dictionary that contains the moderation response.

        Example:
            {
                "text_moderation": {
                    "content": "This is a test",
                    "response": {
                        "sexual": 0.1
                    },
                    "flagged": True
                },
                "nsfw_moderation": {
                    "image_hash": "1234567890",
                    "response": {
                        "IsImageAdultClassified": True
                    },
                    "flagged": True
                },
                "csam_moderation": {
                    "image_hash": "1234567890",
                    "response": {
                        "IsMatch": True
                    },
                    "flagged": True
                }
            }
        """
        print("moderating text: ", text)
        self.reset_moderation_flags()
        text_flagged_map = self.text_moderation_filter(text, model_list, do_moderation)

        if image is not None:
            image_flagged_map = self.image_moderation_filter(image)
        else:
            image_flagged_map = self._NON_TOXIC_IMAGE_MODERATION_MAP

        res = {}
        res.update(text_flagged_map)
        res.update(image_flagged_map)

        return res
