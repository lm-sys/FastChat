import argparse
import code
import datetime
import json
import os
from pytz import timezone
import time

import pandas as pd
from tqdm import tqdm


def get_log_files(max_num_files=None):
    dates = []
    for month in [4, 5]:
        for day in range(1, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")

    num_servers = 12
    filenames = []
    for d in dates:
        for i in range(num_servers):
            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    filenames = filenames[-max_num_files:]
    return filenames


def load_log_files(log_files):
    data = []
    for filename in tqdm(log_files, desc="read files"):
        for retry in range(5):
            try:
                lines = open(filename).readlines()
                break
            except FileNotFoundError:
                time.sleep(2)

        for l in lines:
            row = json.loads(l)

            data.append(
                dict(
                    type=row["type"],
                    tstamp=row["tstamp"],
                    model=row.get("model", ""),
                    models=row.get("models", ["", ""]),
                )
            )

    return data


def merge_counts(series, on, names):
    ret = pd.merge(series[0], series[1], on=on)
    for i in range(2, len(series)):
        ret = pd.merge(ret, series[i], on=on)
    ret = ret.reset_index()
    old_names = list(ret.columns)[-len(series) :]
    rename = {old_name: new_name for old_name, new_name in zip(old_names, names)}
    ret = ret.rename(columns=rename)
    return ret


def report_basic_stats(log_files):
    df_all = load_log_files(log_files)
    df_all = pd.DataFrame(df_all)
    now_t = df_all["tstamp"].max()

    df_1_hour = df_all[df_all["tstamp"] > (now_t - 3600)]
    df_1_day = df_all[df_all["tstamp"] > (now_t - 3600 * 24)]

    # Model call counts
    model_hist_all = df_all[df_all["type"] == "chat"]["model"].value_counts()
    model_hist_1_day = df_1_day[df_1_day["type"] == "chat"]["model"].value_counts()
    model_hist_1_hour = df_1_hour[df_1_hour["type"] == "chat"]["model"].value_counts()
    model_hist = merge_counts(
        [model_hist_all, model_hist_1_day, model_hist_1_hour],
        on="model",
        names=["All", "Last Day", "Last Hour"],
    )
    model_hist_md = model_hist.to_markdown(index=False, tablefmt="github")

    # Action counts
    action_hist_all = df_all["type"].value_counts()
    action_hist_1_day = df_1_day["type"].value_counts()
    action_hist_1_hour = df_1_hour["type"].value_counts()
    action_hist = merge_counts(
        [action_hist_all, action_hist_1_day, action_hist_1_hour],
        on="type",
        names=["All", "Last Day", "Last Hour"],
    )
    action_hist_md = action_hist.to_markdown(index=False, tablefmt="github")

    # Anony vote counts
    anony_vote_df_all = df_all[
        df_all["type"].isin(["leftvote", "rightvote", "tievote", "bothbad_vote"])
    ]
    anony_vote_df_all = anony_vote_df_all[
        anony_vote_df_all["models"].apply(lambda x: x[0] == "")
    ]
    anony_vote_hist_all = anony_vote_df_all["type"].value_counts()

    anony_vote_df_1_day = df_1_day[
        df_1_day["type"].isin(["leftvote", "rightvote", "tievote", "bothbad_vote"])
    ]
    anony_vote_df_1_day = anony_vote_df_1_day[
        anony_vote_df_1_day["models"].apply(lambda x: x[0] == "")
    ]
    anony_vote_hist_1_day = anony_vote_df_1_day["type"].value_counts()

    anony_vote_df_1_hour = df_1_hour[
        df_1_hour["type"].isin(["leftvote", "rightvote", "tievote", "bothbad_vote"])
    ]
    anony_vote_df_1_hour = anony_vote_df_1_hour[
        anony_vote_df_1_hour["models"].apply(lambda x: x[0] == "")
    ]
    anony_vote_hist_1_hour = anony_vote_df_1_hour["type"].value_counts()

    anony_vote_hist = merge_counts(
        [anony_vote_hist_all, anony_vote_hist_1_day, anony_vote_hist_1_hour],
        on="type",
        names=["All", "Last Day", "Last Hour"],
    )
    anony_vote_hist_md = anony_vote_hist.to_markdown(index=False, tablefmt="github")

    # Last 24 hours
    chat_1_day = df_1_day[df_1_day["type"] == "chat"]
    num_chats_last_24_hours = []
    base = df_1_day["tstamp"].min()
    for i in range(24, 0, -1):
        left = base + (i - 1) * 3600
        right = base + i * 3600
        num = ((chat_1_day["tstamp"] >= left) & (chat_1_day["tstamp"] < right)).sum()
        num_chats_last_24_hours.append(num)
    times = [
        datetime.datetime.fromtimestamp(
            base + i * 3600, tz=timezone("US/Pacific")
        ).strftime("%Y-%m-%d %H:%M:%S %Z")
        for i in range(24, 0, -1)
    ]
    last_24_hours_df = pd.DataFrame({"time": times, "value": num_chats_last_24_hours})
    last_24_hours_md = last_24_hours_df.to_markdown(index=False, tablefmt="github")

    # code.interact(local=locals())

    return {
        "model_hist_md": model_hist_md,
        "action_hist_md": action_hist_md,
        "anony_vote_hist_md": anony_vote_hist_md,
        "num_chats_last_24_hours": last_24_hours_md,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    basic_stats = report_basic_stats(log_files)

    print(basic_stats["model_hist_md"] + "\n")
    print(basic_stats["action_hist_md"] + "\n")
    print(basic_stats["anony_vote_hist_md"] + "\n")
    print(basic_stats["num_chats_last_24_hours"] + "\n")
