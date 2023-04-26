import glob
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px

LOG_DIR = "/Users/Ying/work/project/chatbot/server_logs"

# Loading Data
dfs = []
for filename in glob.glob(os.path.join(LOG_DIR, "*.json")):
    with open(filename, "r") as f:
        lines = f.readlines()
    dfs.append(pd.DataFrame([json.loads(l) for l in lines]))
df = pd.concat(dfs).reset_index(drop=True)
print("====== logs overview ======")
print(df.columns)
print(df['type'].value_counts())

# Extracting Battles
battles = (
    df[df['type'].isin(["rightvote", "leftvote", "tievote"])]
    .reset_index(drop=True)
    .drop(columns=["model", "gen_params", "start", "finish", "state"])
    .copy())
battles
# Cleaning up the Models
model_pairs = (
    battles['models']
    .str.join(" vs ")
    .str.replace("<h3>|</h3>|Model A: |Model B: |Model A|Model B|Left: |Right: |\n", "", regex=True)
)
model_pairs
## Address missing models
missing_models = model_pairs == " vs "
np.sum(missing_models)
def extract_models(state):
    return state[0]["model_name"] + " vs " + state[1]["model_name"]

model_pairs[missing_models] = battles[missing_models]['states'].apply(extract_models)
model_pairs
battles['battle'] = model_pairs
battles.drop(columns=["Left", "Right"], inplace=True, errors="ignore")
battles = battles.join(model_pairs.str.split(" vs ", expand=True).set_axis(["Left", "Right"], axis=1))
battles
# Plotting Counts of outcomes for various battles
px.histogram(battles, x="battle", color="type", barmode="group", height=800)
# Estimate Elo Scores
# Normalizing the data by converting the ties to a win for both models
def eliminate_ties(battles):
    # convert ties to pairs of victories
    ties = battles[battles['type'] == "tievote"]
    leftwins = ties.copy().reset_index(drop=True)
    leftwins['type'] = 'leftvote'
    rightwins = ties.copy().reset_index(drop=True)
    rightwins['type'] = 'rightvote'
    no_ties = battles[battles['type'] != "tievote"]
    battles_no_ties = pd.concat([no_ties, leftwins, rightwins])
    return battles_no_ties

def compute_elo(battles):
    battles_no_ties = eliminate_ties(battles)
    
    models = pd.concat([battles_no_ties['Left'], battles_no_ties['Right']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = battles_no_ties.shape[0]
    
    X = np.zeros([n, p])
    X[np.arange(n), models[battles_no_ties['Left']]] = +1
    X[np.arange(n), models[battles_no_ties['Right']]] = -1
    
    Y = np.zeros(n)
    Y[battles_no_ties['type'] == "leftvote"] = 1.0

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)
    
    return pd.DataFrame({"model": models.index, "weight": lr.coef_[0]}).sort_values("weight", ascending=False)

scores = compute_elo(battles)
scores
px.bar(scores, x="model", y="weight")
# Bootstrap Error bars
rows = []
for n in range(5000):
    df = compute_elo(battles.sample(frac=1, replace=True))
    rows.append(df.set_index("model")["weight"])
    
df = pd.DataFrame(rows)
df
px.box(df.melt(), x="model", y="value")
def plot_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        score = df.quantile(.5),
        upper = df.quantile(.975))).reset_index().sort_values("score", ascending=False)
    bars['error_y'] = bars['upper'] - bars["score"]
    bars['error_y_minus'] = bars['score'] - bars["lower"]
    return px.scatter(bars, x="model", y="score", error_y="error_y", 
                      error_y_minus="error_y_minus", 
                      title=title, height = 600)
plot_bootstrap_scores(df, "Bootstrap of Elo Estimates")
# Comparing Pairs

battles['score'] = (battles['type'] == "leftvote") + 0.5 * (battles['type'] == "tievote")
score_pairs = pd.pivot_table(battles, values = "score", index="Left", columns ="Right", aggfunc = "mean")
score_pairs
pd.pivot_table(battles, index="Left", columns ="Right", aggfunc = "size",fill_value=0)
# Treating Battles Symmetrically 
sym = battles[["type", "Left", "Right"]]
sym = pd.concat([
    sym,
    pd.DataFrame({
        "type": sym['type'].map({"leftvote": "rightvote", "tievote": "tievote", "rightvote": "leftvote"}),
        "Left": sym["Right"], "Right": sym["Left"]})
    ]).reset_index(drop=True)
sym
pd.pivot_table(sym, index="Left", columns ="Right", aggfunc = "size", fill_value=0)
px.bar(compute_elo(sym), x="model", y="weight")
rows = []
for n in range(1000):
    df = compute_elo(sym.sample(frac=1, replace=True))
    rows.append(df.set_index("model")["weight"])
    
df = pd.DataFrame(rows)
plot_bootstrap_scores(df, "ELO for All Battles with Bootstrap CI")
# Sampling Battles Evenly
n_per_battle = 5

rows = []
sym_groups = sym.groupby(["Left", "Right"], as_index=False)
for n in range(1000):
    resampled = (sym_groups
                 .apply(lambda grp: grp.sample(n_per_battle, replace=True))
                 .reset_index(drop=True))
    row = compute_elo(resampled)
    rows.append(row.set_index("model")["weight"])
    
df = pd.DataFrame(rows)


plot_bootstrap_scores(df, f"ELO for {n_per_battle} Battles from Each Combination")
n_per_battle = 10

rows = []
sym_groups = sym.groupby(["Left", "Right"], as_index=False)
for n in range(1000):
    resampled = (sym_groups
                 .apply(lambda grp: grp.sample(n_per_battle, replace=True))
                 .reset_index(drop=True))
    row = compute_elo(resampled)
    rows.append(row.set_index("model")["weight"])
    
df = pd.DataFrame(rows)


plot_bootstrap_scores(df, f"ELO for {n_per_battle} Battles from Each Combination")
n_per_battle = 50

rows = []
sym_groups = sym.groupby(["Left", "Right"], as_index=False)
for n in range(1000):
    resampled = (sym_groups
                 .apply(lambda grp: grp.sample(n_per_battle, replace=True))
                 .reset_index(drop=True))
    row = compute_elo(resampled)
    rows.append(row.set_index("model")["weight"])
    
df = pd.DataFrame(rows)


plot_bootstrap_scores(df, f"ELO for {n_per_battle} Battles from Each Combination")



def eliminate_ties(battles):
    # convert ties to pairs of victories
    ties = battles[battles['type'] == "tievote"]
    leftwins = ties.copy().reset_index(drop=True)
    leftwins['type'] = 'leftvote'
    rightwins = ties.copy().reset_index(drop=True)
    rightwins['type'] = 'rightvote'
    no_ties = battles[battles['type'] != "tievote"]
    battles_no_ties = pd.concat([no_ties, leftwins, rightwins])
    return battles_no_ties

def compute_elo(battles):
    battles_no_ties = eliminate_ties(battles)
    
    models = pd.concat([battles_no_ties['Left'], battles_no_ties['Right']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = battles_no_ties.shape[0]
    
    X = np.zeros([n, p])
    X[np.arange(n), models[battles_no_ties['Left']]] = +1
    X[np.arange(n), models[battles_no_ties['Right']]] = -1
    
    Y = np.zeros(n)
    Y[battles_no_ties['type'] == "leftvote"] = 1.0

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)
    
    elo_scores = 400*lr.coef_[0] + 1000
    
    return pd.DataFrame({"model": models.index, "weight": elo_scores}).sort_values("weight", ascending=False)

def plot_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        score = df.quantile(.5),
        upper = df.quantile(.975))).reset_index().sort_values("score", ascending=False)
    bars['error_y'] = bars['upper'] - bars["score"]
    bars['error_y_minus'] = bars['score'] - bars["lower"]
    return px.scatter(bars, x="model", y="score", error_y="error_y", 
                      error_y_minus="error_y_minus", 
                      title=title, height = 600)

# rows = []
# for n in range(1000):
#     df = compute_elo(battles.sample(frac=1, replace=True))
#     rows.append(df.set_index("model")["weight"])
#     
# df = pd.DataFrame(rows)
# df
# 
# px.violin(df.melt(), x="model", y="value")

