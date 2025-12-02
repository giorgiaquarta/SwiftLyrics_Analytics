import nltk
import re
import pandas as pd

# download wordsets only once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def get_ignore_words() -> set:
    """Returns a set of words to ignore."""

    custom_ignore = {
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
        "ah",
        "ha",
        "cause",
    }
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return stop_words.union(custom_ignore)


def clean_text(text: str) -> str:
    """Cleans a single string of text."""

    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\(\[].*?[\)\]]", "", text)  # remove brackets
    text = text.replace("\\", "").replace("-", " ").replace("\n", " ")
    text = re.sub(r"\b\w*'\w+\b", "", text)  # remove words with apostrophes
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    ignore = get_ignore_words()

    # filter
    valid_tokens = [w for w in tokens if w not in ignore and len(w) > 1]
    return " ".join(valid_tokens)


def clean_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    """Applies text cleaning to the 'lyric' column of the dataframe."""

    if "lyric" not in df.columns:
        raise ValueError("DataFrame must contain a 'lyric' column")

    df_clean = df.copy()

    df_clean["lyric_clean"] = df_clean["lyric"].apply(clean_text)
    return df_clean