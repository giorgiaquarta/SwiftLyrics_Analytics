import pandas as pd
import os


def parse_df(
    filename: str, headers: list[str] = None, header=None, delimiter: str = ";"
) -> pd.DataFrame:
    """Generic CSV parser wrapper."""

    raw = None

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        file_path = os.path.join(current_dir, "dataset", filename)

        raw = pd.read_csv(file_path, delimiter=delimiter, header=header, names=headers)
    except Exception as exc:
        print(f"Error loading {filename}: {exc}")

    return raw


def parse_lyrics(filename: str = "lyrics.csv") -> pd.DataFrame:
    """Parses and groups lyrics by album and track."""

    raw_lyrics = parse_df(
        filename, ["album_id", "track_id", "line_n", "part", "lyric"], delimiter=":"
    )
    if raw_lyrics.empty:
        return raw_lyrics

    lyrics = (
        raw_lyrics.groupby(["album_id", "track_id"])["lyric"]
        .apply(lambda lines: "\n".join(lines))
        .reset_index()
    )
    return lyrics


def parse_albums(filename: str = "albums.csv") -> pd.DataFrame:
    """Parses album metadata and drops statistical columns."""

    raw_albums = parse_df(filename, header="infer")
    if raw_albums.empty:
        return raw_albums

    drop_cols = [
        "SubTitle",
        "LowestFqWord",
        "PrevalentVerb",
        "PrevalentAdjective",
        "PrevalentNoun",
        "Songs",
        "Lines",
        "Words",
    ]

    # drop columns that actually exist
    existing_drop_cols = [c for c in drop_cols if c in raw_albums.columns]
    albums = raw_albums.drop(existing_drop_cols, axis=1)
    return albums