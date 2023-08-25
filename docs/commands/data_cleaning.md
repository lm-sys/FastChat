## Data cleaning

## Requirements
```
pip3 install bs4 markdownify
pip3 install polyglot pyicu pycld2
```

## Steps
```
# Convert html to markdown
python3 -m fastchat.data.clean_sharegpt --in sharegpt_html.json --out sharegpt_clean.json

# Keep or remove specific languages
python3 -m fastchat.data.optional_clean --in sharegpt_clean.json --out sharegpt_clean_lang.json --skip-lang SOME_LANGUAGE_CODE

# Split long conversations
python3 -m fastchat.data.split_long_conversation --in sharegpt_clean_lang.json --out sharegpt_clean_lang_split.json --model-name /home/ubuntu/model_weights/llama-7b/
```
