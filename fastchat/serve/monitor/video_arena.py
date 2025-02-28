"""Video Arena module for displaying video model leaderboard and rankings.

This module provides functionality to fetch, process and display leaderboard data
from the Video Arena API. It includes model information, score processing, and
Gradio UI components for visualization.

Classes:
    ModelVersion: Data class for storing model version information
    ModelInfo: Data class for storing model class information

Functions:
    process_video_arena_leaderboard: Process raw leaderboard data into DataFrame
    build_video_arena_tab: Build and display the Video Arena leaderboard UI
"""

import os
from dataclasses import dataclass
import requests
import pandas as pd
import gradio as gr

from fastchat.serve.monitor.monitor import recompute_final_ranking

# URL for fetching Video Arena leaderboard data
VIDEO_ARENA_LEADERBOARD_URL = os.getenv(
    "VIDEO_ARENA_LEADERBOARD_URL", "https://www.videoarena.tv/api/v1/leaderboard"
)


@dataclass
class ModelVersion:
    """Data class to store model version information.

    Attributes:
        license: The license type of the model version
        website: The website URL for the model version
    """

    license: str
    website: str


@dataclass
class ModelInfo:
    """Data class to store model class information.

    Attributes:
        official_name: The official display name of the model
        organization: The organization that created the model
        versions: Dictionary mapping version names to ModelVersion objects
    """

    official_name: str
    organization: str
    versions: dict[str, ModelVersion] = None


# Mapping of model identifiers to their information
VIDEO_MODEL_INFO = {
    "veo": ModelInfo(
        "Veo",
        "Google",
        versions={
            "2.0": ModelVersion(
                "Proprietary", "https://deepmind.google/technologies/veo/veo-2/"
            )
        },
    ),
    "minimax": ModelInfo(
        "Minimax",
        "Hailuo",
        versions={"01": ModelVersion("Proprietary", "https://hailuoai.video/")},
    ),
    "kling": ModelInfo(
        "Kling",
        "Kuaishou",
        versions={
            "1.0": ModelVersion("Proprietary", "https://klingai.com/"),
            "1.5": ModelVersion("Proprietary", "https://klingai.com/"),
        },
    ),
    "sora": ModelInfo(
        "Sora",
        "OpenAI",
        versions={"1": ModelVersion("Proprietary", "https://openai.com/sora/")},
    ),
    "luma": ModelInfo(
        "Luma",
        "LumaLabs",
        versions={
            "1.6": ModelVersion("Proprietary", "https://lumalabs.ai/dream-machine")
        },
    ),
    "runway": ModelInfo(
        "Runway",
        "Runway",
        versions={
            "default": ModelVersion(
                "Proprietary", "https://runwayml.com/research/introducing-gen-3-alpha"
            )
        },
    ),
    "genmo": ModelInfo(
        "Genmo",
        "Genmo",
        versions={
            "0.2": ModelVersion("Proprietary", "https://www.genmo.ai/"),
            "Mochi-1": ModelVersion("Apache-2.0", "https://www.genmo.ai/"),
        },
    ),
    "svd": ModelInfo(
        "SVD",
        "StabilityAI",
        versions={
            "1.0": ModelVersion("Proprietary", "https://stability.ai/stable-video")
        },
    ),
    "opensora": ModelInfo(
        "OpenSora",
        "OpenSora",
        versions={
            "1.2": ModelVersion("Apache-2.0", "https://github.com/hpcaitech/Open-Sora")
        },
    ),
    "pika": ModelInfo(
        "Pika",
        "PikaLabs",
        versions={
            "β": ModelVersion("Proprietary", "pika.art"),
            "1.5": ModelVersion("Proprietary", "pika.art"),
        },
    ),
}


def process_video_arena_leaderboard(data):
    """Process raw leaderboard data into a formatted DataFrame.

    Args:
        data: Raw leaderboard data from the Video Arena API containing model scores
             and metadata

    Returns:
        pd.DataFrame: Processed leaderboard with rankings and confidence intervals.
                     Contains columns for rank, model name, scores, confidence intervals,
                     vote counts, organization and license information.
    """
    leaderboard = []
    for item in data:
        model_name = item["model"].lower()
        version = item["version"]

        # Skip veo model
        if model_name == "veo":
            continue

        # Get model info from mapping
        model_info = VIDEO_MODEL_INFO.get(
            model_name,
            ModelInfo(
                model_name, "Unknown", {"default": ModelVersion("Proprietary", "")}
            ),
        )

        # Determine license and website based on version
        version_key = version if version in model_info.versions else "default"
        if version_key not in model_info.versions:
            model_info.versions["default"] = ModelVersion("Proprietary", "")
        model_version = model_info.versions[version_key]
        license_type, website = model_version.license, model_version.website

        # Replace spaces with dashes in the display name
        display_name = f"{model_info.official_name} {version}".replace(" ", "-")
        model_data = {
            "name": f"[{display_name}]({website})" if website else display_name,
            "visibility": "public",
            "score": round(item["scores"]["elo"]),
            "lower": item["scores"]["elo"] - item["scores"]["ci_lower"],
            "upper": item["scores"]["elo"] + item["scores"]["ci_upper"],
            "votes": (
                item["scores"]["win"]["total"]
                + item["scores"]["loss"]["total"]
                + item["scores"]["tie"]["total"]
            ),
            "organization": model_info.organization,
            "license": license_type,
        }
        leaderboard.append(model_data)

    leaderboard = pd.DataFrame(leaderboard)

    # Calculate confidence intervals
    leaderboard["rating_q975"] = leaderboard["upper"].round().astype(int)
    leaderboard["rating_q025"] = leaderboard["lower"].round().astype(int)

    leaderboard["upper_diff"] = leaderboard["upper"] - leaderboard["score"]
    leaderboard["lower_diff"] = leaderboard["score"] - leaderboard["lower"]

    # Round the differences to integers
    leaderboard["upper_diff"] = leaderboard["upper_diff"].round().astype(int)
    leaderboard["lower_diff"] = leaderboard["lower_diff"].round().astype(int)

    leaderboard["confidence_interval"] = (
        "+"
        + leaderboard["upper_diff"].astype(str)
        + "/-"
        + leaderboard["lower_diff"].astype(str)
    )

    # Calculate rankings using the existing function
    rankings_ub = recompute_final_ranking(leaderboard)
    leaderboard.insert(loc=0, column="Rank* (UB)", value=rankings_ub)

    # Sort the leaderboard
    leaderboard = leaderboard.sort_values(
        by=["Rank* (UB)", "score"], ascending=[True, False]
    )

    return leaderboard


def build_video_arena_tab():
    """Build and display the Video Arena leaderboard tab in the Gradio interface.

    Fetches data from the Video Arena API, processes it into a formatted DataFrame,
    and creates a formatted display using Gradio components including:
    - Summary statistics (number of models and battles)
    - Interactive leaderboard table
    - Explanatory text for ranking methodology

    Returns:
        None. Creates and displays Gradio UI components directly.
    """
    response = requests.get(VIDEO_ARENA_LEADERBOARD_URL)
    if response.status_code == 200:
        leaderboard = process_video_arena_leaderboard(response.json())
        leaderboard = leaderboard.rename(
            columns={
                "name": "Model",
                "confidence_interval": "95% CI",
                "score": "Arena Score",
                "organization": "Organization",
                "votes": "Votes",
                "license": "License",
            }
        )

        column_order = [
            "Rank* (UB)",
            "Model",
            "Arena Score",
            "95% CI",
            "Votes",
            "Organization",
            "License",
        ]
        leaderboard = leaderboard[column_order]
        num_models = len(leaderboard)
        total_battles = int(leaderboard["Votes"].sum()) // 2

        md = f"""
        [VideoArena](https://www.videoarena.tv/) is a free AI video service allowing users to access, compare, and rank
        text-to-video capabilities of state-of-the-art generative models. This
        leaderboard contains the relative performance and ranking of {num_models}
        models over {total_battles} battles.
        """

        gr.Markdown(md, elem_id="leaderboard_markdown")
        gr.DataFrame(
            leaderboard,
            datatype=["number", "markdown", "number", "str", "number", "str", "str"],
            elem_id="video_arena_leaderboard",
            height=600,
            wrap=True,
            interactive=False,
            column_widths=[70, 130, 60, 80, 50, 80, 70],
        )

        gr.Markdown(
            """
    ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models
    that are statistically better than the target model.
    Model A is statistically better than model B when A's lower-bound score is greater
    than B's upper-bound score (in 95% confidence interval). \n
    **Confidence Interval**: represents the range of uncertainty around the Arena Score.
    It's displayed as +X / -Y, where X is the difference between the upper bound and
    the score, and Y is the difference between the score and the lower bound.
    """,
            elem_id="leaderboard_markdown",
        )
    else:
        gr.Markdown("Error with fetching Video Arena data. Check back in later.")
