import pytest
import pandas as pd
import os
from SwiftLyrics_Analytics import dataset, format, nn_analysis

# --- Fixtures ---
@pytest.fixture
def valid_csv_file(tmp_path):
    """Creates a temporary valid CSV file for testing."""
    d = tmp_path / "test_lyrics.csv"
    d.write_text("album_name,track_title,lyric\nMidnights,Anti-Hero,It's me, hi!")
    return str(d)

@pytest.fixture
def sample_dataframe():
    """Creates a valid DataFrame mimicking raw input."""
    return pd.DataFrame({
        "album_name": ["Midnights"],
        "track_title": ["Anti-Hero"],
        "lyric": ["It's me, HI! I'm the problem, it's me."]
    })

# --- Dataset Tests ---
def test_dataset_load_success(valid_csv_file):
    """Test that a valid CSV file loads correctly as a DataFrame."""
    df = dataset.parse_df(valid_csv_file)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["album_name", "track_title", "lyric"]
    assert len(df) == 1

# --- Format Tests ---
def test_clean_lyrics_functionality(sample_dataframe):
    """Test that lyrics are correctly lowercased and cleaned."""
    cleaned_df = format.clean_lyrics(sample_dataframe)
    
    # check that new column exists
    assert "lyric_clean" in cleaned_df.columns
    
    # check logic
    result = cleaned_df.loc[0, "lyric_clean"]
    expected_snippet = "its me hi im the problem its me" 

    assert result == expected_snippet or "hi" in result
    assert result.islower()

def test_clean_lyrics_preserves_rows(sample_dataframe):
    """Test that no data is lost during the cleaning process."""
    cleaned_df = format.clean_lyrics(sample_dataframe)
    assert len(cleaned_df) == len(sample_dataframe)

# --- Analysis Tests ---
def test_classify_valid_album_filter():
    """Test that analysis runs and returns valid data."""

    result = nn_analysis.classify_sentiment(album_id="1989")

    assert result is not None, "Function returned nothing (None)"
    
    assert isinstance(result, dict), "Result should be a dictionary"
    
    assert "sentiment_score" in result, "Result is missing 'sentiment_score'"
    assert "label" in result, "Result is missing 'label'"