import pandas as pd


def parse_df(
    filename: str, headers: list[str] = None, header=None, delimiter: str = ";"
) -> pd.DataFrame:
    raw = None
    try:
        raw = pd.read_csv(
            f"../dataset/{filename}", delimiter=delimiter, header=header, names=headers
        )
    except Exception as exc:
        print(exc)

    return raw


def parse_lyrics(filename: str) -> pd.DataFrame:
    raw_lyrics = parse_df(
        filename, ["album_id", "track_id", "line_n", "part", "lyric"], delimiter=":"
    )
    lyrics = (
        raw_lyrics.groupby(["album_id", "track_id"])["lyric"]
        .apply(lambda lines: "\n".join(lines))
        .reset_index()
    )

    return lyrics


def parse_albums(filename: str) -> pd.DataFrame:
    raw_albums = parse_df(
        filename,
        header="infer",
    )
    albums = raw_albums.drop(
        [
            "SubTitle",
            "LowestFqWord",
            "PrevalentVerb",
            "PrevalentAdjective",
            "PrevalentNoun",
            "Songs",
            "Lines",
            "Words",
        ],
        axis=1,
    )

    return albums


if __name__ == "__main__":
    lyrics = parse_lyrics("lyrics.csv")
    print(lyrics.head(20))

    albums = parse_albums("albums.csv")
    print(albums.head(13))