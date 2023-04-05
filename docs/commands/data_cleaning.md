### Data cleanning
```
python3 -m fastchat.data.clean_sharegpt --in sharegpt_20230322_html.json --out sharegpt_20230322_clean.json
python3 -m fastchat.data.optional_clean --in sharegpt_20230322_clean.json --out sharegpt_20230322_clean_lang.json --skip-lang
python3 -m fastchat.data.split_long_conversation --in sharegpt_20230322_clean_lang.json --out sharegpt_20230322_clean_lang_split.json --model-name /home/ubuntu/model_weights/hf-llama-7b/
```
