import nltk
import re
import string
import pandas as pd


def remove_words(data: str):
    data = data.str.replace(r"[\(\[].*?[\)\]]", "")
    data = data.str.replace("\\", "")
    data = data.str.replace("-", " ")
    data = data.apply(lambda x: re.sub(r"\b\w*'\w+\b", "", x))
    data = data.apply(lambda x: re.sub(r"[^\w\s]", "", x))
    data = data.str.replace("\n", " ")
    data = data.str.lower()
    data = data.str.replace("[{}]".format(string.punctuation), "")

    clean_text = list()
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ignore_words = [
        "yeah",
        "ya",
        "na",
        "wan",
        "uh",
        "gon",
        "ima",
        "mm",
        "uhhuh",
        "bout",
        "em",
        "nigga",
        "niggas",
        "got",
        "ta",
        "lil",
        "ol",
        "hey",
        "oooh",
        "ooh",
        "oh",
        "youre",
        "dont",
        "im",
        "youve",
        "ive",
        "theres",
        "ill",
        "yaka",
        "lalalala",
        "la",
        "da",
        "di",
        "yuh",
        "shawty",
        "oohooh",
        "shoorah",
        "mmmmmm",
        "ook",
        "bidibambambambam",
        "shh",
        "bro",
        "ho",
        "aint",
        "cant",
        "know",
        "bambam",
        "shitll",
        "tonka",
        "ah ah",
        "ah",
        "oh",
        "oh oh",
        "di di",
        "di",
        "uh huh",
        "ooh ooh",
        "ha",
        "cause",
    ]
    ignore = nltk.corpus.stopwords.words("english").copy() + ignore_words.copy()

    for i in data:
        words = nltk.word_tokenize(i)
        for element in ignore:  # given the tokenized list, return a list that doesn't contain any of the elements
            words = list(filter(lambda x: x != element and len(x) > 1, words))
        lyric = " ".join(words)
        clean_text.append(lyric)

    return clean_text


def clean_lyrics(df: pd.DataFrame):
    df_clean = df.copy()
    df_clean.loc[:, "lyric_clean"] = remove_words(df.loc[:, "lyric"])
    return df_clean


if __name__ == "__main__":
    from dataset import parse_lyrics

    lyrics = parse_lyrics("lyrics.csv")
    print(clean_lyrics(lyrics).head())