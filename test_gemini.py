import base64
from io import BytesIO
import os
import requests
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
)


def encode_image_base64(image_path):
    """Encode an image in base64."""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return base64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return base64.b64encode(image_path).decode("utf-8")
    else:
        # image_path is PIL.WebPImagePlugin.WebPImageFile
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_image(image_file):
    from PIL import Image

    image = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image

def main():
    model_name = "gemini-1.0-pro-vision"
    temperature = 0.5
    top_p = 0.7
    max_new_tokens=1024

    dog_image_path = "/home/ec2-user/sglang/examples/quick_start/images/cat.jpeg"
    picsum_photo = "https://picsum.photos/200/300"
    image = encode_image_base64(load_image("https://picsum.photos/200/300"))

    project_id = os.environ.get("GCP_PROJECT_ID", None)
    location = os.environ.get("GCP_LOCATION", None)
    vertexai.init(project=project_id, location=location)

    # generator = GenerativeModel(model_name).generate_content(
    #     messages,
    #     stream=True,
    #     generation_config=GenerationConfig(top_p=top_p, max_output_tokens=max_new_tokens, temperature=temperature),
    # )

    # first_turn = ""
    # for ret in generator:
    #     first_turn += ret.text

    print("First turn done.")

    messages = []
    vertexai_message = {"role": "user", "parts": [{"text": "What's in this image?"}]}
    vertexai_message["parts"].append({"inline_data" : {"data": image, "mime_type": "image/jpeg"}})
    messages.append(vertexai_message)
    messages.append({"role": "model", "parts": [{"text": "this is an image about grass."}]})
    messages.append({"role": "user", "parts": [{"text": "tell me about alpacas."}]})

    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    print(f"==== request ====\n{gen_params}")
    
    generator = GenerativeModel(model_name).generate_content(
        messages,
        stream=True,
        generation_config=GenerationConfig(top_p=top_p, max_output_tokens=max_new_tokens, temperature=temperature),
    )

    for ret in generator:
        data = {
            "text" : ret.text
        }

        yield data


if __name__ == "__main__":
    generator = main()
    for chunk in generator:
        print(chunk["text"])