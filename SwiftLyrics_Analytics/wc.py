from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from dataset import parse_albums, parse_lyrics
from format import clean_lyrics


def gen_wordcloud(df: pd.DataFrame, bg="white", cm="winter"):
    wc = WordCloud(collocations=False, background_color=bg, colormap=cm).generate(
        " ".join(df)
    )
    return wc


def display_wordcloud(df: pd.DataFrame):
    wc = gen_wordcloud(df)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def display_album_wordcloud(album_df: pd.DataFrame, lyric_df: pd.DataFrame):
    index = 1
    album = album_df["Code"][:-1]
    plt.figure(figsize=(15, 15))
    for a in album:
        d = lyric_df[lyric_df["album_id"] == a]
        color = "#" + album_df.loc[album_df["Code"] == a, "Color"].values[0]
        wordcloud = gen_wordcloud(d["lyric_clean"], bg=color)
        plt.subplot(4, 3, index)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(album_df.loc[album_df["Code"] == a, "Title"].iloc[0])
        index += 1
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    from dataset import parse_lyrics
    from format import clean_lyrics

    lyrics = clean_lyrics(parse_lyrics("lyrics.csv"))
    # display_wordcloud(lyrics["lyric_clean"])

    albums = parse_albums("albums.csv")
    display_album_wordcloud(albums, lyrics)