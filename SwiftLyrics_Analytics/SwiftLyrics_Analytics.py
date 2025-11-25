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
    - analysis LYRICS_FILENAME: perform a neural network analysis of the dataset
"""


def dataset_command(lyrics, albums):
    lyrics = dataset.parse_lyrics(lyrics)
    albums = dataset.parse_albums(albums)
    print(lyrics.head())
    print(albums.head())


def format_command(lyrics):
    lyrics_df = dataset.parse_lyrics(lyrics)
    clean_lyrics_df = format.clean_lyrics(lyrics_df)
    print(clean_lyrics_df.head())


def wordcloud_command(lyrics, albums):
    lyrics_df = format.clean_lyrics(dataset.parse_lyrics(lyrics))
    wc.display_wordcloud(lyrics_df["lyric_clean"])

    albums_df = dataset.parse_albums(albums)
    wc.display_album_wordcloud(albums_df, lyrics_df)


def analysis_command(lyrics, album=None, track=None):
    lyrics_df = dataset.parse_lyrics(lyrics)

    classified_df = nn_analysis.classify(lyrics_df, album, track)
    print(classified_df.head(20))

def parse_args_to_kwargs(args_list):
    """Parses a list of strings 'key=value' into a dictionary."""
    kwargs = {}
    for arg in args_list:
        if "=" in arg:
            # Split only on the first '=' to allow '=' in values
            key, value = arg.split("=", 1)
            # Optional: Strip dashes if user types --key=value
            key = key.lstrip('-') 
            kwargs[key] = value
    return kwargs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python script.py <command> [key=value ...]")

    command = sys.argv[1]
    # Grab everything after the command
    raw_args = sys.argv[2:] 
    
    # Convert list to dictionary
    kwargs = parse_args_to_kwargs(raw_args)

    match command:
        case "dataset":
            # Unpack the dictionary into the function
            # User types: python main.py dataset input=data.csv output=clean.csv
            dataset_command(**kwargs) 
        case "format":
            # User types: python main.py format file=text.txt
            format_command(**kwargs)
        case "wordcloud":
            wordcloud_command(**kwargs)
        case "analysis":
            analysis_command(**kwargs)
        case _:
            sys.exit("Unknown command.")