"""
Usage:
python3 -m unittest tests.test_image_utils
"""

import base64
from io import BytesIO
import os
import unittest

import numpy as np
from PIL import Image

from fastchat.utils import (
    resize_image_and_return_image_in_bytes,
    image_moderation_filter,
)
from fastchat.conversation import get_conv_template


def check_byte_size_in_mb(image_base64_str):
    return len(image_base64_str) / 1024 / 1024


def generate_random_image(target_size_mb, image_format="PNG"):
    # Convert target size from MB to bytes
    target_size_bytes = target_size_mb * 1024 * 1024

    # Estimate dimensions
    dimension = int((target_size_bytes / 3) ** 0.5)

    # Generate random pixel data
    pixel_data = np.random.randint(0, 256, (dimension, dimension, 3), dtype=np.uint8)

    # Create an image from the pixel data
    img = Image.fromarray(pixel_data)

    # Save image to a temporary file
    temp_filename = "temp_image." + image_format.lower()
    img.save(temp_filename, format=image_format)

    # Check the file size and adjust quality if needed
    while os.path.getsize(temp_filename) < target_size_bytes:
        # Increase dimensions or change compression quality
        dimension += 1
        pixel_data = np.random.randint(
            0, 256, (dimension, dimension, 3), dtype=np.uint8
        )
        img = Image.fromarray(pixel_data)
        img.save(temp_filename, format=image_format)

    return img


class DontResizeIfLessThanMaxTest(unittest.TestCase):
    def test_dont_resize_if_less_than_max(self):
        max_image_size = 5
        initial_size_mb = 0.1  # Initial image size
        img = generate_random_image(initial_size_mb)

        image_bytes = BytesIO()
        img.save(image_bytes, format="PNG")  # Save the image as JPEG
        previous_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        image_bytes = resize_image_and_return_image_in_bytes(
            img, max_image_size_mb=max_image_size
        )
        new_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        self.assertEqual(previous_image_size, new_image_size)


class ResizeLargeImageForModerationEndpoint(unittest.TestCase):
    def test_resize_large_image_and_send_to_moderation_filter(self):
        initial_size_mb = 6  # Initial image size which we know is greater than what the endpoint can take
        img = generate_random_image(initial_size_mb)

        nsfw_flag, csam_flag = image_moderation_filter(img)
        self.assertFalse(nsfw_flag)
        self.assertFalse(nsfw_flag)


class DontResizeIfMaxImageSizeIsNone(unittest.TestCase):
    def test_dont_resize_if_max_image_size_is_none(self):
        initial_size_mb = 0.2  # Initial image size
        img = generate_random_image(initial_size_mb)

        image_bytes = BytesIO()
        img.save(image_bytes, format="PNG")  # Save the image as JPEG
        previous_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        image_bytes = resize_image_and_return_image_in_bytes(
            img, max_image_size_mb=None
        )
        new_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        self.assertEqual(previous_image_size, new_image_size)


class OpenAIConversationDontResizeImage(unittest.TestCase):
    def test(self):
        conv = get_conv_template("chatgpt")
        initial_size_mb = 0.2  # Initial image size
        img = generate_random_image(initial_size_mb)
        image_bytes = BytesIO()
        img.save(image_bytes, format="PNG")  # Save the image as JPEG
        previous_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        resized_img = conv.convert_image_to_base64(img)
        resized_img_bytes = base64.b64decode(resized_img)
        new_image_size = check_byte_size_in_mb(resized_img_bytes)

        self.assertEqual(previous_image_size, new_image_size)


class ClaudeConversationResizesCorrectly(unittest.TestCase):
    def test(self):
        conv = get_conv_template("claude-3-haiku-20240307")
        initial_size_mb = 5  # Initial image size
        img = generate_random_image(initial_size_mb)
        image_bytes = BytesIO()
        img.save(image_bytes, format="PNG")  # Save the image as JPEG
        previous_image_size = check_byte_size_in_mb(image_bytes.getvalue())

        resized_img = conv.convert_image_to_base64(img)
        new_base64_image_size = check_byte_size_in_mb(resized_img)
        new_image_bytes_size = check_byte_size_in_mb(base64.b64decode(resized_img))

        self.assertLess(new_image_bytes_size, previous_image_size)
        self.assertLessEqual(new_image_bytes_size, conv.max_image_size_mb)
        self.assertLessEqual(new_base64_image_size, 5)
