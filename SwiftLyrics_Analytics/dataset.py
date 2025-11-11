import pandas as pd

def parse_df(filename: str, headers: list[str], delimiter: str = ';') -> pd.DataFrame :

    raw = None
    try: 
        raw = pd.read_csv(
            f'dataset/{filename}',
            delimiter=':',
            header=None,
            names=headers
        )
    except Exception as exc:
        print(exc)

    return raw

def parse_lyrics(filename: str) -> pd.DataFrame:
    raw_lyrics = parse_df(filename, ['album_id', 'track_id', 'line_n', 'part', 'lyric'])
    lyrics = raw.lyrics.groupby(['album_id','track_id'])['lyric'].apply(lambda lines: "\n".join(lines)).reset_index()

    return lyrics

def parse_albums(filename: str) -> pd.DataFrame:
    raw.albums = parse_df(filename, [''])

if __name__ = "__main__"
    lyrics = parse_lyrics("lyrics.csv")
    print(lyrics.head(20))