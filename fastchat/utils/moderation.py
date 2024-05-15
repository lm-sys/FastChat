"""
Moderation utilities.
"""
import io
import os
import time
import requests


def oai_moderation(text, custom_thresholds=None):
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
                    if getattr(res.results[0].category_scores, category) > threshold:
                        flagged = True
            break
        except (openai.OpenAIError, KeyError, IndexError) as e:
            print(f"MODERATION ERROR: {e}\nInput: {text}")
    return flagged


def pil_image_to_byte_array(pil_image, format="JPEG"):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def convert_image_to_byte_array(image):
    if type(image) == str:
        with open(image, "rb") as image_data:
            image_bytes = image_data.read()
    else:
        image_bytes = pil_image_to_byte_array(
            image, format="JPEG"
        )  # Use 'PNG' or other formats as needed

    return image_bytes


def moderation_filter(text, model_list, do_moderation=False):
    # Apply moderation for below models
    MODEL_KEYWORDS = ["claude", "gpt", "bard", "mistral-large", "command-r", "dbrx"]

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
        return oai_moderation(text, custom_thresholds)
    return False


def image_moderation_request(image, endpoint, api_key):
    print(f"moderating image: {image}")

    headers = {"Content-Type": "image/jpeg", "Ocp-Apim-Subscription-Key": api_key}

    # Specify the API URL
    image_bytes = convert_image_to_byte_array(image)
    MAX_RETRIES = 3
    for _ in range(MAX_RETRIES):
        response = requests.post(endpoint, headers=headers, data=image_bytes).json()
        if response["Status"] == 3000:
            break
        else:
            time.sleep(0.5)

    return response


def image_moderation_provider(image, api_type):
    if api_type == "nsfw":
        endpoint = os.environ["AZURE_IMG_MODERATION_ENDPOINT"]
        api_key = os.environ["AZURE_IMG_MODERATION_API_KEY"]
        response = image_moderation_request(image, endpoint, api_key)
        return response["IsImageAdultClassified"] or response["IsImageRacyClassified"]
    elif api_type == "csam":
        endpoint = (
            "https://api.microsoftmoderator.com/photodna/v1.0/Match?enhance=false"
        )
        api_key = os.environ["PHOTODNA_API_KEY"]
        response = image_moderation_request(image, endpoint, api_key)
        return response["IsMatch"]


def image_moderation_filter(image):
    nsfw_flagged = image_moderation_provider(image, "nsfw")
    csam_flagged = image_moderation_provider(image, "csam")
    return nsfw_flagged, csam_flagged
