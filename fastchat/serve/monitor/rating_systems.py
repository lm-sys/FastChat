import os
import math
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm


STYLE_CONTROL_ELEMENTS_V1 = [
    "sum_assistant_a_tokens",
    "header_count_a",
    "list_count_a",
    "bold_count_a",
    "sum_assistant_b_tokens",
    "header_count_b",
    "list_count_b",
    "bold_count_b",
]


def get_matchups_models(df):
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_elo(df):
    """
    in Elo we want numpy arrays for matchups and outcomes
      matchups: int32 (N,2)  contains model ids for the competitors in a match
      outcomes: float64 (N,) contains 1.0, 0.5, or 0.0 representing win, tie, or loss for model_a
    """
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def preprocess_for_bt(df):
    """in BT we only need the unique (matchup,outcome) sets along with the weights of how often they occur"""
    n_rows = len(df)
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    # set the two model cols by mapping the model names to their int ids
    schedule[:, [0, 1]], models = get_matchups_models(df)
    # map outcomes to integers (must be same dtype as model ids so it can be in the same array)
    # model_a win -> 2, tie -> 1 (prefilled by default), model_b win -> 0
    schedule[df["winner"] == "model_a", 2] = 2
    schedule[df["winner"] == "model_b", 2] = 0
    # count the number of occurances of each observed result
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    # map 2 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used as labels during optimization
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # each possible result is weighted according to number of times it occured in the dataset
    return matchups, outcomes, models, weights


def preprocess_for_style(
    df,
    apply_ratio=[1, 1, 1, 1],
    style_elements=STYLE_CONTROL_ELEMENTS_V1,
    add_one=True,
):
    matchups, outcomes, models = preprocess_for_elo(
        df
    )  # this can use the same preprocessing as Elo

    n = matchups.shape[0]
    k = int(len(style_elements) / 2)

    def extract_style_feature(x, feature):
        val = x[feature]
        if isinstance(val, int):
            return val
        else:
            return sum(val.values())

    style_vector = np.zeros(shape=(2 * k, n), dtype=np.int32)
    for idx, element in enumerate(style_elements):
        style_vector[idx, :] = df.conv_metadata.map(
            partial(extract_style_feature, feature=element)
        ).values
    style_vector = np.ascontiguousarray(style_vector)

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    # Apply ratio where necessary (length, etc)
    style_diff[apply_ratio] /= style_sum[apply_ratio]

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return matchups, features, outcomes, models


def fit_vectorized_elo(
    matchups,
    outcomes,
    sample_indices,
    num_models,
    k=4.0,
    base=10.0,
    init_rating=1000.0,
    scale=400.0,
):
    """fit multiple sets of Elo ratings on different samples of the data at the same time"""
    alpha = math.log(base) / scale
    num_samples = sample_indices.shape[1]
    ratings = np.zeros(shape=(num_samples, num_models), dtype=np.float64)
    # iterate over the rows of sample_indices, each column is an index into a match in the input arrays
    sample_range = np.arange(num_samples)
    for matchup_indices in sample_indices:
        model_a_indices = matchups[matchup_indices, 0]
        model_b_indices = matchups[matchup_indices, 1]
        model_a_ratings = ratings[sample_range, model_a_indices]
        model_b_ratings = ratings[sample_range, model_b_indices]
        sample_outcomes = outcomes[matchup_indices]
        probs = expit(alpha * (model_a_ratings - model_b_ratings))
        updates = k * (sample_outcomes - probs)
        ratings[sample_range, model_a_indices] += updates
        ratings[sample_range, model_b_indices] -= updates
    return ratings + init_rating


def compute_elo(df, k=4.0, base=10.0, init_rating=1000.0, scale=400.0):
    matchups, outcomes, models = preprocess_for_elo(df)
    alpha = math.log(base) / scale
    ratings = np.full(shape=(len(models),), fill_value=init_rating)
    for (model_a_idx, model_b_idx), outcome in zip(matchups, outcomes):
        prob = 1.0 / (
            1.0 + math.exp(alpha * (ratings[model_b_idx] - ratings[model_a_idx]))
        )
        update = k * (outcome - prob)
        ratings[model_a_idx] += update
        ratings[model_b_idx] -= update
    return {model: ratings[idx] for idx, model in enumerate(models)}


def compute_bootstrap_elo(
    df, num_round=100, k=4.0, base=10.0, init_rating=1000.0, scale=400.0
):
    matchups, outcomes, models = preprocess_for_elo(df)
    sample_indices = np.random.randint(low=0, high=len(df), size=(len(df), num_round))
    ratings = fit_vectorized_elo(
        matchups, outcomes, sample_indices, len(models), k, base, init_rating, scale
    )
    df = pd.DataFrame(data=ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    # this form naturally counts a draw as half a win and half a loss
    loss = -(
        (np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights
    ).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    # aggregate gradients at the model level using the indices in matchups
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def scale_and_offset(
    ratings,
    models,
    scale=400,
    init_rating=1000,
    baseline_model="mixtral-8x7b-instruct-v0.1",
    baseline_rating=1114,
):
    """convert ratings from the natural scale to the Elo rating scale with an anchored baseline"""
    scaled_ratings = (ratings * scale) + init_rating
    if baseline_model in models:
        baseline_idx = models.index(baseline_model)
        scaled_ratings += baseline_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings


def compute_bt(df, base=10.0, scale=400.0, init_rating=1000, tol=1e-6):
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base), tol)
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating=init_rating)
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles,
    num_round,
    base=10.0,
    scale=400.0,
    init_rating=1000.0,
    tol=1e-6,
    num_cpu=None,
):
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles), pvals=weights / weights.sum(), size=(num_round)
    )
    # only the distribution over their occurance counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(battles)

    # the only thing different across samples is the distribution of weights
    bt_fn = partial(
        fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(base), tol=tol
    )
    with mp.Pool(num_cpu if num_cpu else os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(bt_fn, boot_weights), total=num_round))

    ratings = np.array(results)
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating)
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


DIFF_MASK = np.array(
    [1.0, -1.0], dtype=np.float64
)  # create globally to not incur the instantiation cost in each call


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    reg_loss = half_reg * np.inner(params, params)

    # Split params into ratings and feature parameters
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )

    error = outcomes - probs
    grad = reg * params  # initialize the grad as the regularization grad
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)
    return loss, grad


# note on regularization:
# default reg is to 0.5 since the LogisticRegression default is 1.0
# in the original implementation, matchups were duplicated
# that made the ratio of log loss to reg loss "twice as high"
# in this non-duplicated version for parity we also reduce the reg by one half to match
def fit_contextual_bt(
    matchups,
    features,
    outcomes,
    models,
    idxs=None,
    alpha=math.log(10.0),
    reg=0.5,
    tol=1e-6,
):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    # sample idxs optionally allow for fitting on a bootstrap sample of the dataset
    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df, alpha=math.log(10.0), reg=0.5, init_rating=1000.0, scale=400.0, tol=1e-6
):
    matchups, features, outcomes, models = preprocess_for_style(df)
    ratings_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = ratings_params[: len(models)]
    params = ratings_params[len(models) :]
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating)
    scaled_ratings = pd.Series(scaled_ratings, index=models).sort_values(
        ascending=False
    )
    return scaled_ratings, params


def compute_bootstrap_style_control(
    df,
    num_round,
    alpha=math.log(10.0),
    reg=0.5,
    init_rating=1000.0,
    scale=400.0,
    tol=1e-6,
    num_cpu=None,
):
    matchups, features, outcomes, models = preprocess_for_style(df)

    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )

    boot_idxs = np.random.randint(
        low=0, high=matchups.shape[0], size=(num_round, matchups.shape[0])
    )

    with mp.Pool(num_cpu if num_cpu else os.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap_unordered(contextual_bt_fn, boot_idxs), total=num_round)
        )

    ratings_params = np.array(results)
    ratings = ratings_params[:, : len(models)]
    params = ratings_params[:, len(models) :]
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating)
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index], params
