"""
Convert a leaderboard csv file to html table used in the blog.

Usage:
python3 leaderboard_csv_to_html.py --in leaderboard_table_20230619.csv
"""
import argparse

import numpy as np

from fastchat.serve.monitor.monitor import load_leaderboard_table_csv


def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}"> {model_name} </a>'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    data = load_leaderboard_table_csv(args.input, add_hyperlink=False)
    headers = [
        "Model",
        "MT-bench (score)",
        "Arena Elo rating",
        "MMLU",
        "License",
    ]
    values = []
    for item in data:
        row = []
        for key in headers:
            value = item[key]
            row.append(value)
        row[0] = model_hyperlink(item["Model"], item["Link"])
        values.append(row)
    values.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 1e9)

    for value in values:
        row = "<tr>"
        for x in value:
            try:
                if np.isnan(x):
                    x = "-"
            except TypeError:
                pass
            row += f" <td>{x}</td> "
        row += "</tr>"
        print(row)
