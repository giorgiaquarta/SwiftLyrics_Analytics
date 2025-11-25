import dataset
import format
import wc
import nn_analysis
import sys

usage = """
USAGE: SwiftLyrics_Analytics.py COMMAND

COMMAND may be one of the following:
    - dataset LYRICS_FILENAME ALBUM_FILENAME: print the heading lines of the dataset
    - format LYRICS_FILENAME: clean the lyrics and print the data frame
    - wordcloud LYRICS_FILENAME ALBUM_FILENAME: display the wordcloud of the dataset
    - analysis LYRICS_FILENAME ALBUM_FILENAME: perform a neural network analysis of the dataset
"""


def dataset_command(lyrics, albums):
    lyrics = dataset.parse_lyrics(lyrics)
    albums = dataset.parse_albums(albums)
    print(lyrics.head())
    print(albums.head())


def format_command(lyrics):
    lyrics_df = datset.parse_lyrics(lyrics)
    clean_lyrics_df = format.clean_lyrics(lyrics_df)
    print("clean_lyrics_df.head()")


def wordcloud_command(lyrics, albums):
    lyrics_df = format.clean_lyrics(dataset.parse_lyrics(dataset.parse_lyrics(lyrics)))
    wc.display_wordcloud(lyrics_df["lyric_clean"])

    albums_df = dataset.parse_albums(albums)
    wc.display_album_wordcloud(albums_df, lyrics_df)


def analysis_command(lyrics, albums):
    print("pippo")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(usage)

    match sys.argv[1]:
        case "dataset":
            dataset_command(sys.argv[2], sys.argv[3])
        case "format":
            format_command(sys.argv[2])
        case "wordcloud":
            wordcloud_command(sys.argv[2], sys.argv[3])
        case "analysis":
            analysis_command(sys.argv[2], sys.argv[3])
        case _:
            sys.exit(usage)