from fastchat.serve.logstats import print_query_rate
from fastchat.serve.rating import print_ratings, print_rating_algo

LOG_DIR = "/home/Ying/arena_logs"
MARKDOWN = "leaderboard.md"


if __name__ == "__main__":
    with open(MARKDOWN, "w") as f:
        print_ratings(LOG_DIR, f)
        print_query_rate(LOG_DIR, f)
        print_rating_algo(f)

