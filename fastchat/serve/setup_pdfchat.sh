#!/usr/bin/env bash

# Install Python packages
pip install llama-index-core llama-parse llama-index-readers-file python-dotenv
pip install polyglot
pip install PyICU
pip install pycld2
pip install pytesseract

pip install pdf2image

# Clone the Tesseract tessdata repository
git clone https://github.com/tesseract-ocr/tessdata

# cd into tessdata and set TESSDATA_PREFIX to the current directory
cd tessdata
export TESSDATA_PREFIX="$(pwd)"

echo "TESSDATA_PREFIX is set to: $TESSDATA_PREFIX"