import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np


def gen_wordcloud(
    text_series: pd.Series, bg: str = "white", cm: str = "winter"
) -> WordCloud:
    """
    Generates a WordCloud object from a series of text strings.

    Args:
        text_series (pd.Series): strings to include in the wordcloud.
        bg (str): background color hex code or name.
        cm (str): colormap name (matplotlib colormap).

    Returns:
        WordCloud: The generated wordcloud object.
    """
    #join text, ensure strings, drop NAs
    text = " ".join(text_series.dropna().astype(str).tolist())

    # generate wordcloud
    wc = WordCloud(
        collocations=False, background_color=bg, colormap=cm, width=800, height=400
    ).generate(text)

    return wc


def display_wordcloud(text_series: pd.Series):
    """
    Generates and displays a single wordcloud plot for the provided text.

    Args:
        text_series (pd.Series): the data to visualize.
    """
    # Terminal Output
    print("Showing figure 1/2: most frequent words in Taylor Swift's discography...")

    wc = gen_wordcloud(text_series)
     # Window Title (via 'num') and Figure setup
    plt.figure(num="Figure 1/2: Complete Discography", figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
     # add a visible title on the chart itself
    plt.title("Most frequent words in Taylor Swift's discography")
    plt.show()


def display_album_wordcloud(album_df: pd.DataFrame, lyric_df: pd.DataFrame):
    """
    Generates and displays a grid of wordclouds, one for each album.

    Args:
        album_df (pd.DataFrame): DataFrame containing album metadata.
                                 Expected columns: 'Code', 'Color', 'Title'.
        lyric_df (pd.DataFrame): DataFrame containing lyrics.
                                 Expected columns: 'album_id', 'lyric_clean' (or 'lyric').
    """
    # Terminal Output
    print("Showing figure 2/2: most frequent words per album...")

    n_albums = len(album_df)
    cols = 3
    rows = int(np.ceil(n_albums / cols))

    # Window Title (via 'num')
    plt.figure(num="Figure 2/2: Album Details", figsize=(15, 5 * rows))

    # Add a main title to the entire grid
    plt.suptitle("Most frequent words per album", fontsize=20)

    for i, (_, album_row) in enumerate(album_df.iterrows(), 1):
        album_code = album_row["Code"]

        # ensure color format is valid
        color_hex = str(album_row.get("Color", "000000"))
        if not color_hex.startswith("#"):
            color_hex = "#" + color_hex

        # filter lyrics for this album
        album_lyrics = lyric_df[lyric_df["album_id"] == album_code]

        if album_lyrics.empty:
            continue

        try:
            # check if column exists, fallback to 'lyric' if 'lyric_clean' missing
            wc = gen_wordcloud(album_lyrics["lyric_clean"], bg=color_hex)

            plt.subplot(rows, cols, i)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(album_row.get("Title", f"Album {album_code}"))
        except ValueError:
            pass

    plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.95) # Adjusted top to make room for suptitle
    plt.show()
