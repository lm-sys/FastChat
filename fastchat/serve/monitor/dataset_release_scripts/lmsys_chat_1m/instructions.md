```
export BASE=clean_conv_20230809_100k_pii
export SCALE=10

# filter words
python3 filter_bad_conv.py --in $BASE.json

# Clean up some fileds (e.g., timestamps)
python3 final_post_processing.py --in $BASE.s1.json

# upload to hf
python3 upload_hf_dataset.py --in $BASE.s1.s2.json

# Make another version with openai moderation tag
python3 merge_oai_tag.py --in $BASE.s1.s2.json

# Make visualizations
python3 compute_stats.py --in $BASE.s1.json --scale $SCALE

# Copy figures
scp "atlas:/data/lmzheng/FastChat/fastchat/serve/monitor/dataset_release_scripts/lmsys_chat_1m/*.pdf" .
```

