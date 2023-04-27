"""
pip3 install scikit-learn plotly tabulate
"""
import argparse

from fastchat.serve.monitor.logstats import print_query_rate
from fastchat.serve.monitor.elo_rating_mle import print_ratings_mle, print_rating_mle_algo
from fastchat.serve.monitor.elo_rating_linear_update import (
        print_ratings_linear_update, print_rating_linear_update_algo)


def get_log_files():
    dates = ["2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27"]
    num_servers = 10

    filenames = []
    for i in range(num_servers):
        for d in dates:
            filenames.append(f"/home/ubuntu/fastchat_logs/server{i}/{d}-conv.json")

    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=str,
        default="leader.md")
    args = parser.parse_args()

    log_files = get_log_files()

    with open(args.out_file, "w") as f:
        # Ratings
        print_ratings_mle(log_files, f)
        print_ratings_linear_update(log_files, f)

        # Print algorithm description
        print_rating_mle_algo(f)
        print_rating_linear_update_algo(f)
