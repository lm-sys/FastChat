"""
From colab:
https://colab.research.google.com/drive/1oMdw_Lqgmd6DletSOLHsyD-Rc96cRShs?usp=sharing
"""
import argparse
import datetime
import json
import os
from pytz import timezone
import time

import kaleido
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

import plotly.io as pio

pio.kaleido.scope.mathjax = None

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, required=True)
parser.add_argument("--scale", type=int, required=True)
args = parser.parse_args()

filename = args.in_file
scale = args.scale
convs = json.load(open(filename))
df = pd.DataFrame(convs)
df

print(f"#ips: {df['user_id'].nunique() * scale}")
print(f"#models: {df['model'].nunique()}")
print(f"#language: {df['language'].nunique()}")
print(f"#turns: {df['turn'].mean()}")

model_counts = df["model"].value_counts() * scale
# print("model counts", model_counts)
fig = px.bar(x=model_counts.index, y=model_counts)
fig.update_layout(
    xaxis_title=None,
    yaxis_title="Count",
    height=200,
    width=950,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()
fig.write_image("model_count.pdf")


model_counts = df["language"].value_counts().head(25) * scale
fig = px.bar(x=model_counts.index, y=model_counts)
fig.update_layout(
    xaxis_title=None,
    yaxis_title="Count",
    height=200,
    width=950,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()
fig.write_image("language_count.pdf")

chat_dates = [
    datetime.datetime.fromtimestamp(x, tz=timezone("US/Pacific")).strftime("%Y-%m-%d")
    for x in df["tstamp"]
]


def to_remove(x):
    for d in ["08-09", "08-08", "08-07", "08-06", "08-05", "08-04"]:
        if d in x:
            return True
    return False


chat_dates = [x for x in chat_dates if not to_remove(x)]

chat_dates_counts = pd.value_counts(chat_dates) * scale
print(f"mean #chat per day: {np.mean(chat_dates_counts):.2f}")

fig = px.bar(x=chat_dates_counts.index, y=chat_dates_counts)
fig.update_layout(
    xaxis_title="Dates",
    yaxis_title="Count",
    height=200,
    width=950,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()
fig.write_image("daily_conversation_count.pdf")

import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "lmsys/vicuna-7b-v1.5", use_fast=False
)

prompts = []
responses = []
for conv in df["conversation"]:
    for row in conv:
        if row["role"] == "user":
            prompts.append(row["content"])
        else:
            responses.append(row["content"])

print(f"#prompts: {len(prompts)}")
print(f"#responses: {len(responses)}")


prompt_lens = [len(tokenizer(x).input_ids) for x in tqdm(prompts)]
print()
print(f"mean prompt len: {np.mean(prompt_lens):.2f}")

response_lens = [len(tokenizer(x).input_ids) if x else 0 for x in tqdm(responses)]
print()
print(f"mean response len: {np.mean(response_lens):.2f}")
