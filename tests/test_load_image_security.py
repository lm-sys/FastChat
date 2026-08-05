"""
Usage:
python3 -m unittest tests.test_load_image_security
"""

import unittest
from unittest.mock import patch

from fastchat.utils import _is_safe_http_url, load_image


class LoadImageSecurityTest(unittest.TestCase):
    def test_blocks_private_http_urls(self) -> None:
        self.assertFalse(_is_safe_http_url("http://127.0.0.1/image.png"))
        self.assertFalse(_is_safe_http_url("http://169.254.169.254/latest/meta-data/"))

    def test_allows_public_http_urls(self) -> None:
        self.assertTrue(_is_safe_http_url("https://example.com/image.png"))

    @patch("fastchat.utils.requests.get")
    def test_load_image_rejects_internal_url(self, mock_get) -> None:
        with self.assertRaises(ValueError):
            load_image("http://127.0.0.1/secret.png")
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
