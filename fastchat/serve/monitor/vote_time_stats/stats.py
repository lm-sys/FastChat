import argparse

import matplotlib.pyplot as plt
import pandas as pd


def compute_statistics(df):
    # Calculate the total count of each language
    language_counts = df['language'].value_counts(normalize=True)

    # Determine the languages that account for at least 0.1% of the dataset
    major_languages = language_counts[language_counts >= 0.001].index

    # Filter the DataFrame to include only major languages
    df_major_languages = df[df['language'].isin(major_languages)]

    # Group by language and calculate the median time_to_vote
    median_time_to_vote = df_major_languages.groupby('language')['time_to_vote'].median()
    median_time_to_vote = median_time_to_vote.sort_values()

    plt.figure(figsize=(10, 6))
    median_time_to_vote.plot(kind='bar', color='skyblue')
    plt.xlabel('Language')
    plt.ylabel('Median Time to Vote')
    plt.title('Median Time to Vote per Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot to a file with high resolution
    plt.savefig(f"distribution_time_to_vote_language.png", dpi=300)

    # Group by language and calculate the median time_to_vote
    median_time_to_vote = df_major_languages.groupby('language')['time_to_vote_per_char'].median()
    median_time_to_vote = median_time_to_vote.sort_values()

    plt.figure(figsize=(10, 6))
    median_time_to_vote.plot(kind='bar', color='skyblue')
    plt.xlabel('Language')
    plt.ylabel('Median Time to Vote')
    plt.title('Median Time to Vote per Language (per character)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot to a file with high resolution
    plt.savefig(f"distribution_time_to_vote_per_char_language.png", dpi=300)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute other statistics")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file", default=None)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    compute_statistics(df)
