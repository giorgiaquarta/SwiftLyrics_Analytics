import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def classify(
    df: pd.DataFrame, album_id: Optional[str] = None, track_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs zero-shot classification on lyrics to determine sentiment, theme, and style.

    Args:
        df (pd.DataFrame): DataFrame containing lyrics. Must contain 'lyric', 'album_id', and 'track_id'.
        album_id (str, optional): filter for specific album ID.
        track_id (str, optional): filter for specific track ID.

    Returns:
        pd.DataFrame: a new DataFrame with classification columns appended.
    """

    import torch
    from transformers import pipeline

    # filter data for album or track identifier
    filtered_df = df.copy()
    if album_id:
        filtered_df = filtered_df[filtered_df["album_id"] == album_id]
    if track_id:
        filtered_df = filtered_df[filtered_df["track_id"].astype(str) == str(track_id)]

    if filtered_df.empty:
        logger.warning("No data found for the specified album/track criteria.")
        return pd.DataFrame(
            columns=["album_id", "track_id", "sentiment", "theme", "style"]
        )

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading pipelines on device {device}...")

    try:
        sentiment_classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            hypothesis_template="the mood of this song is {}.",
            device=device,
        )

        theme_classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            hypothesis_template="this song is about {}.",
            device=device,
            multi_label=True,
        )

        style_classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            hypothesis_template="This text is written in a {} style.",
            device=device,
        )

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    sentiment_labels = ["positive", "negative", "neutral"]
    theme_labels = [
        "love",
        "grief",
        "family",
        "friendship",
        "power",
        "self-worth",
        "revenge",
    ]
    style_labels = ["aulic", "medium", "simple"]

    results = []
    total_songs = len(filtered_df)

    logger.info(f"Starting classification for {total_songs} songs...")

    for i, (index, row) in enumerate(filtered_df.iterrows()):
        song_text = row.get("lyric", "")

        # skip empty lyrics
        # if not isinstance(song_text, str) or not song_text.strip():
        #     continue

        logger.info(f"Processing song {i + 1}/{total_songs}")

        try:
            # classification
            sent_res = sentiment_classifier(
                song_text, candidate_labels=sentiment_labels
            )
            theme_res = theme_classifier(song_text, candidate_labels=theme_labels)
            style_res = style_classifier(song_text, candidate_labels=style_labels)

            results.append(
                {
                    "album_id": row["album_id"],
                    "track_id": row["track_id"],
                    "sentiment": sent_res["labels"][0],
                    "theme": theme_res["labels"][0],
                    "style": style_res["labels"][0],
                }
            )
        except Exception as e:
            logger.error(
                f"Error classifying {row.get('album_id', 'Unknown')}:{row.get('track_id', 'Unknown')} - {e}"
            )

    logger.info("Classification complete.")
    return pd.DataFrame(results)
