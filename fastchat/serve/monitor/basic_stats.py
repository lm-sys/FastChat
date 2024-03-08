import argparse
import code
import datetime
import json
import os
from pytz import timezone
import time

import pandas as pd  # pandas>=2.0.3
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


NUM_SERVERS = 14
LOG_ROOT_DIR = "~/fastchat_logs"


def get_log_files(max_num_files=None):
    log_root = os.path.expanduser(LOG_ROOT_DIR)
    filenames = []
    for i in range(NUM_SERVERS):
        for filename in os.listdir(f"{log_root}/server{i}"):
            if filename.endswith("-conv.json"):
                filepath = f"{log_root}/server{i}/{filename}"
                name_tstamp_tuple = (filepath, os.path.getmtime(filepath))
                filenames.append(name_tstamp_tuple)
    # sort by tstamp
    filenames = sorted(filenames, key=lambda x: x[1])
    filenames = [x[0] for x in filenames]

    max_num_files = max_num_files or len(filenames)
    filenames = filenames[-max_num_files:]
    return filenames


def load_log_files(filename):
    data = []
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


def load_log_files_parallel(log_files, num_threads=16):
    data_all = []
    from multiprocessing import Pool

    with Pool(num_threads) as p:
        ret_all = list(tqdm(p.imap(load_log_files, log_files), total=len(log_files)))
        for ret in ret_all:
            data_all.extend(ret)
    return data_all


def get_anony_vote_df(df):
    anony_vote_df = df[
        df["type"].isin(["leftvote", "rightvote", "tievote", "bothbad_vote"])
    ]
    anony_vote_df = anony_vote_df[anony_vote_df["models"].apply(lambda x: x[0] == "")]
    return anony_vote_df


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
    df_all = load_log_files_parallel(log_files)
    df_all = pd.DataFrame(df_all)
    now_t = df_all["tstamp"].max()
    df_1_hour = df_all[df_all["tstamp"] > (now_t - 3600)]
    df_1_day = df_all[df_all["tstamp"] > (now_t - 3600 * 24)]
    anony_vote_df_all = get_anony_vote_df(df_all)

    # Chat trends
    chat_dates = [
        datetime.datetime.fromtimestamp(x, tz=timezone("US/Pacific")).strftime(
            "%Y-%m-%d"
        )
        for x in df_all[df_all["type"] == "chat"]["tstamp"]
    ]
    chat_dates_counts = pd.value_counts(chat_dates)
    vote_dates = [
        datetime.datetime.fromtimestamp(x, tz=timezone("US/Pacific")).strftime(
            "%Y-%m-%d"
        )
        for x in anony_vote_df_all["tstamp"]
    ]
    vote_dates_counts = pd.value_counts(vote_dates)
    chat_dates_bar = go.Figure(
        data=[
            go.Bar(
                name="Anony. Vote",
                x=vote_dates_counts.index,
                y=vote_dates_counts,
                text=[f"{val:.0f}" for val in vote_dates_counts],
                textposition="auto",
            ),
            go.Bar(
                name="Chat",
                x=chat_dates_counts.index,
                y=chat_dates_counts,
                text=[f"{val:.0f}" for val in chat_dates_counts],
                textposition="auto",
            ),
        ]
    )
    chat_dates_bar.update_layout(
        barmode="stack",
        xaxis_title="Dates",
        yaxis_title="Count",
        height=300,
        width=1200,
    )

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
    anony_vote_hist_all = anony_vote_df_all["type"].value_counts()
    anony_vote_df_1_day = get_anony_vote_df(df_1_day)
    anony_vote_hist_1_day = anony_vote_df_1_day["type"].value_counts()
    # anony_vote_df_1_hour = get_anony_vote_df(df_1_hour)
    # anony_vote_hist_1_hour = anony_vote_df_1_hour["type"].value_counts()
    anony_vote_hist = merge_counts(
        [anony_vote_hist_all, anony_vote_hist_1_day],
        on="type",
        names=["All", "Last Day"],
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

    # Last update datetime
    last_updated_tstamp = now_t
    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    # code.interact(local=locals())

    return {
        "chat_dates_bar": chat_dates_bar,
        "model_hist_md": model_hist_md,
        "action_hist_md": action_hist_md,
        "anony_vote_hist_md": anony_vote_hist_md,
        "num_chats_last_24_hours": last_24_hours_md,
        "last_updated_datetime": last_updated_datetime,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    basic_stats = report_basic_stats(log_files)

    print(basic_stats["action_hist_md"] + "\n")
    print(basic_stats["model_hist_md"] + "\n")
    print(basic_stats["anony_vote_hist_md"] + "\n")
    print(basic_stats["num_chats_last_24_hours"] + "\n")
