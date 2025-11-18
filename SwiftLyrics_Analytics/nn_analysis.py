import torch
from transformers import pipeline
import pandas as pd


def classify(df: pd.DataFrame):
    device = 0 if torch.cuda.is_available() else -1

    sentiment_classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        hypothesis_template="the mood of this song is {}.",
        device=device,
    )

    theme_classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        hypothesis_template="this song is about {}.",
        device=device,
        multi_label=True,
    )

    style_classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        hypothesis_template="This text is written in a(n) {} style.",
        device=device,
    )

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
    style_labels = ["poetic", "casual", "quirky"]

    print("----- CLASSIFIER -----")

    classified_df = pd.DataFrame(
        columns=["album_id", "track_id", "sentiment", "theme", "style"]
    )
    df.reset_index()

    for index, row in df.iterrows():
        song = row["lyric"]
        print(f"classifying {row['album_id']}:{row['track_id']}")
        sentiment = sentiment_classifier(song, candidate_labels=sentiment_labels)[
            "labels"
        ][0]
        theme = theme_classifier(song, candidate_labels=theme_labels)["labels"][0]
        style = style_classifier(song, candidate_labels=style_labels)["labels"][0]
        classified_df.loc[index] = [
            row["album_id"],
            row["track_id"],
            sentiment,
            theme,
            style,
        ]

    print("Done!")
    return classified_df


if __name__ == "__main__":
    from dataset import parse_lyrics

    lyrics = parse_lyrics("lyrics.csv")

    classified = classify(lyrics)
    print(classified.head(20))
