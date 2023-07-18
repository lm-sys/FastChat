"""
Upload to huggingface.
"""
import json
from datasets import Dataset, DatasetDict, load_dataset

objs = json.load(open("clean_battle_conv_20230630_tagged_v3_pii_33k_added.json"))
data = Dataset.from_list(objs)
data.push_to_hub("lmsys/chatbot_arena_conversations", private=True)
