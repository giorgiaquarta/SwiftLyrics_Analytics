# SwiftLyrics Analytics

**SwiftLyrics Analytics** is a Python-based tool designed to analyze and compare the themes, sentiment, and lyrics of Taylor Swift's entire discography. It utilizes Natural Language Processing (NLP) and Zero-Shot Classification using Neural Networks to provide deep insights into the lyrics.

## Features

* **Dataset Management**: Parse and view raw lyric and album data.
* **Text Preprocessing**: Clean and tokenize lyrics using NLTK (removing stopwords, punctuation, and custom ignore words).
* **Visualization**: Generate word clouds to visualize the most frequent words globally or per album.
* **Neural Network Analysis**: Perform zero-shot classification using the `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` model to determine:
    * **Sentiment**: Positive, Negative, Neutral.
    * **Theme**: Love, Grief, Family, Friendship, Power, Self-worth, Revenge.
    * **Style**: Aulic, Medium, Simple.

## Installation

### Prerequisites

* Python 3.x
* PIP

### Steps

1.  Clone the repository:
    ```bash
    git clone [https://github.com/giorgiaquarta/SwiftLyrics_Analytics.git](https://github.com/giorgiaquarta/SwiftLyrics_Analytics.git)
    cd SwiftLyrics_Analytics
    ```

2.  Install the package and dependencies:
    ```bash
    pip install .
    ```
    *Dependencies include: `numpy`, `pandas`, `nltk`, `wordcloud`, `matplotlib`, `jupyter`, `transformers`, `torch`.*

3.  (Optional) Install development dependencies for testing:
    ```bash
    pip install .[dev]
    ```

## Usage

Once installed, the project provides a command-line interface (CLI) tool named `swift_lyrics`.

### 1. View Dataset Info
Print the heading lines of the lyrics and albums datasets to verify data loading.
```bash
swift_lyrics dataset
```

### 2. Format and Clean Lyrics
Apply text cleaning algorithms (tokenization, stopword removal) to the lyrics and display the output dataframe.
```bash
swift_lyrics format
```

### 3. Generate Word Clouds

Display a global word cloud of all lyrics and specific word clouds for each album.

```bash
swift_lyrics wordcloud
```

### 4. Neural Network Analysis

Perform zero-shot classification on the lyrics. You can analyze the whole discography or filter by specific Album IDs or Track IDs.

**Run on all lyrics (Warning: computationally expensive):**

```bash
swift_lyrics analysis
```

**Filter by Album (e.g., 'TSW' for Taylor Swift, 'RED' for Red):**

```bash
swift_lyrics analysis --album RED
```

**Filter by specific Track ID:**

```bash
swift_lyrics analysis --album RED --track 01
```

*Note: The first time you run the analysis, it will download the DeBERTa model from Hugging Face.*

## Dataset Codes

The project uses specific codes for albums found in the dataset. Available album codes include:

  * `TSW`: Taylor Swift
  * `FER`: Fearless (Taylor's Version)
  * `SPN`: Speak Now (Taylor's Version)
  * `RED`: Red (Taylor's Version)
  * `NEN`: 1989 (Taylor's Version)
  * `REP`: Reputation
  * `LVR`: Lover
  * `FOL`: Folklore
  * `EVE`: Evermore
  * `MID`: Midnights
  * `TPD`: The Tortured Poets Department
  * `LSG`: The Life Of A Showgirl
  * `OTH`: Other Songs

## Project Structure

```text
SwiftLyrics_Analytics/
├── SwiftLyrics_Analytics/
│   ├── dataset/
│   │   ├── albums.csv       # Album metadata
│   │   └── lyrics.csv       # Lyrics database
│   ├── __init__.py
│   ├── __main__.py          # Entry point script
│   ├── SwiftLyrics_Analytics.py # Main CLI logic
│   ├── dataset.py           # Parsing logic
│   ├── format.py            # NLP cleaning logic
│   ├── nn_analysis.py       # PyTorch/Transformers analysis
│   ├── wc.py                # WordCloud generation
│   └── ...
├── test/                    # Unit tests
├── pyproject.toml           # Project configuration & dependencies
├── setup.py                 # Setup script
└── LICENSE                  # MIT License
```

## Testing

To run the tests, ensure you have the dev dependencies installed: 
```bash
pip install -e .[dev]
```

Run the testing module:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Author

  * **Giorgia Quarta** - *giorgia.quarta3@studio.unibo.it*
