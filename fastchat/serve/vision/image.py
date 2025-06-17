import base64
from enum import auto, IntEnum
from io import BytesIO

from pydantic import BaseModel


class ImageFormat(IntEnum):
    """Image formats."""

    URL = auto()
    LOCAL_FILEPATH = auto()
    PIL_IMAGE = auto()
    BYTES = auto()
    DEFAULT = auto()


class Image(BaseModel):
    url: str = ""
    filetype: str = ""
    image_format: ImageFormat = ImageFormat.BYTES
    base64_str: str = ""

    def convert_image_to_base64(self):
        """Given an image, return the base64 encoded image string."""
        from PIL import Image
        import requests

        # Load image if it has not been loaded in yet
        if self.image_format == ImageFormat.URL:
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert("RGBA")
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
        elif self.image_format == ImageFormat.LOCAL_FILEPATH:
            image = Image.open(self.url).convert("RGBA")
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
        elif self.image_format == ImageFormat.BYTES:
            image_bytes = image

        img_b64_str = base64.b64encode(image_bytes).decode()

        return img_b64_str

    def to_openai_image_format(self):
        if self.image_format == ImageFormat.URL:  # input is a url
            return self.url
        elif self.image_format == ImageFormat.LOCAL_FILEPATH:  # input is a local image
            self.base64_str = self.convert_image_to_base64(self.url)
            return f"data:image/{self.filetype};base64,{self.base64_str}"
        elif self.image_format == ImageFormat.BYTES:
            return f"data:image/{self.filetype};base64,{self.base64_str}"
        else:
            raise ValueError(
                f"This file is not valid or not currently supported by the OpenAI API: {self.url}"
            )

    def resize_image_and_return_image_in_bytes(self, image, max_image_size_mb):
        import math

        image_format = "png"
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 1024, 1024
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if longest_edge != max(image.size):
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        if max_image_size_mb:
            target_size_bytes = max_image_size_mb * 1024 * 1024

            current_size_bytes = image_bytes.tell()
            if current_size_bytes > target_size_bytes:
                resize_factor = (target_size_bytes / current_size_bytes) ** 0.5
                new_width = math.floor(image.width * resize_factor)
                new_height = math.floor(image.height * resize_factor)
                image = image.resize((new_width, new_height))

                image_bytes = BytesIO()
                image.save(image_bytes, format="PNG")
                current_size_bytes = image_bytes.tell()

            image_bytes.seek(0)

        return image_format, image_bytes

    def convert_url_to_image_bytes(self, max_image_size_mb):
        from PIL import Image

        if self.url.endswith(".svg"):
            import cairosvg

            with open(self.url, "rb") as svg_file:
                svg_data = svg_file.read()

            png_data = cairosvg.svg2png(bytestring=svg_data)
            pil_image = Image.open(BytesIO(png_data)).convert("RGBA")
        else:
            pil_image = Image.open(self.url).convert("RGBA")

        image_format, image_bytes = self.resize_image_and_return_image_in_bytes(
            pil_image, max_image_size_mb
        )

        img_base64_str = base64.b64encode(image_bytes.getvalue()).decode()

        return image_format, img_base64_str

    def to_conversation_format(self, max_image_size_mb):
        image_format, image_bytes = self.convert_url_to_image_bytes(
            max_image_size_mb=max_image_size_mb
        )

        self.filetype = image_format
        self.image_format = ImageFormat.BYTES
        self.base64_str = image_bytes

        return self


if __name__ == "__main__":
    image = Image(url="fastchat/serve/example_images/fridge.jpg")
    image.to_conversation_format(max_image_size_mb=5 / 1.5)

    json_str = image.model_dump_json()
    print(json_str)
