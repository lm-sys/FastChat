import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_elo_mle_with_tie(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = pd.pivot_table(
        df[df["winner"].isin(["tie", "tie (bothbad)"])],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = ptbl_tie + ptbl_tie.T
    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Elo scores")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file", default=None)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    elo_series = compute_elo_mle_with_tie(df)

    # Plot the sorted ELO scores
    plt.figure(figsize=(20, 6))
    bars = elo_series.head(20).plot(kind='bar', color='skyblue')
    # Add numbers above each bar
    for bar in bars.patches:
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 10,  # Add a small offset to move the text above the bar
          f'{int(bar.get_height())}',
          ha='center',
          va='bottom',
          fontsize=12,
      )
    plt.xlabel('Model')
    plt.ylabel('ELO Score')
    plt.title('ELO Scores of Models')
    plt.xticks(rotation=60, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"elo_rating_{os.path.splitext(os.path.basename(args.input))[0]}.png", dpi=300)
