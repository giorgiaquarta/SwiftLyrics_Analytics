import argparse
from . import dataset, format, wc, nn_analysis


def dataset_command(args):
    """Handles the 'dataset' command."""

    lyrics = dataset.parse_lyrics(args.lyrics)
    albums = dataset.parse_albums(args.albums)
    print("--- Lyrics Head ---")
    print(lyrics.head())
    print("\n--- Albums Head ---")
    print(albums.head())


def format_command(args):
    """Handles the 'format' command."""

    lyrics_df = dataset.parse_lyrics(args.lyrics)
    clean_lyrics_df = format.clean_lyrics(lyrics_df)
    print(clean_lyrics_df.head())


def wordcloud_command(args):
    """Handles the 'wordcloud' command."""

    lyrics_df = format.clean_lyrics(dataset.parse_lyrics(args.lyrics))
    albums_df = dataset.parse_albums(args.albums)

    print("Displaying global wordcloud...")
    wc.display_wordcloud(lyrics_df["lyric_clean"])

    print("Displaying album-specific wordclouds...")
    wc.display_album_wordcloud(albums_df, lyrics_df)


def analysis_command(args):
    """Handles the 'analysis' command."""

    lyrics_df = dataset.parse_lyrics(args.lyrics)
    classified_df = nn_analysis.classify(lyrics_df, args.album, args.track)
    print(classified_df.head(20))


def main():
    parser = argparse.ArgumentParser(
        description="SwiftLyrics Analytics: analysis of Taylor Swift lyrics."
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Command: dataset
    parser_ds = subparsers.add_parser(
        "dataset", help="Print the heading lines of the dataset"
    )
    parser_ds.add_argument(
        "--lyrics", default="lyrics.csv", help="Filename of lyrics CSV"
    )
    parser_ds.add_argument(
        "--albums", default="albums.csv", help="Filename of albums CSV"
    )
    parser_ds.set_defaults(func=dataset_command)

    # Command: format
    parser_fmt = subparsers.add_parser(
        "format", help="Clean lyrics and print dataframe"
    )
    parser_fmt.add_argument(
        "--lyrics", default="lyrics.csv", help="Filename of lyrics CSV"
    )
    parser_fmt.set_defaults(func=format_command)

    # Command: wordcloud
    parser_wc = subparsers.add_parser("wordcloud", help="Display wordclouds")
    parser_wc.add_argument(
        "--lyrics", default="lyrics.csv", help="Filename of lyrics CSV"
    )
    parser_wc.add_argument(
        "--albums", default="albums.csv", help="Filename of albums CSV"
    )
    parser_wc.set_defaults(func=wordcloud_command)

    # Command: analysis
    parser_an = subparsers.add_parser(
        "analysis", help="Perform neural network analysis"
    )
    parser_an.add_argument(
        "--lyrics", default="lyrics.csv", help="Filename of lyrics CSV"
    )
    parser_an.add_argument("--album", default=None, help="Filter by Album ID")
    parser_an.add_argument("--track", default=None, help="Filter by Track ID")
    parser_an.set_defaults(func=analysis_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()