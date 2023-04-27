import argparse

from fastchat.serve.monitor.logstats import print_query_rate
from fastchat.serve.monitor.mle_rating import print_ratings_mle, print_rating_mle_algo
from fastchat.serve.monitor.elo_rating import (
        print_ratings_linear_update, print_rating_linear_update_algo)

MARKDOWN = "leaderboard.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="../../../../arena_logs")
    args = parser.parse_args()

    with open(MARKDOWN, "w") as f:
        # ratings
        print_ratings_mle(args.log_dir, f)
        print_ratings_linear_update(args.log_dir, f)

        # print algorithm description
        print_rating_mle_algo(f)
        print_rating_linear_update_algo(f)

        # print battle stats

        # print serving stats
        print_query_rate(args.log_dir, f)

