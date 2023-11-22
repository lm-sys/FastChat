export BASE=clean_conv_20230809_1.5M_pii
#export BASE=clean_conv_20230809_100k_pii
export SCALE=1

# Filter words
python3 filter_bad_conv.py --in $BASE.json --sample 1000000

# Clean up some fileds (e.g., timestamps)
python3 final_post_processing.py --in $BASE.s1.json

# Upload to hf
python3 upload_hf_dataset.py --in $BASE.s1.s2.json

# Make another version with openai moderation tag
python3 merge_oai_tag.py --in $BASE.s1.s2.json

# Make visualizations
python3 compute_stats.py --in $BASE.s1.json --scale $SCALE
