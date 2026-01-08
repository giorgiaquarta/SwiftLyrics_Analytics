import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
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
        "album_id": ["1989"],
        "track_id": ["1"],
        "lyric": ["It's me, HI! I'm the problem, it's me."]
    })

# --- Dataset Tests ---
def test_dataset_load_success(valid_csv_file):
    """Test that a valid CSV file loads correctly as a DataFrame."""
    df = dataset.parse_df(valid_csv_file)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    df_str = df.to_string()
    assert "Midnights" in df_str

# --- Format Tests ---
def test_clean_lyrics_functionality(sample_dataframe):
    cleaned = format.clean_lyrics(sample_dataframe)
    assert "lyric_clean" in cleaned.columns
    result = cleaned.loc[0, "lyric_clean"]
    assert "hi" in result or "problem" in result
    assert result.islower()

def test_clean_lyrics_preserves_rows(sample_dataframe):
    cleaned = format.clean_lyrics(sample_dataframe)
    assert len(cleaned) == len(sample_dataframe)

# --- Analysis Tests ---
def test_classify_functionality(sample_dataframe):
    """
    Test that classify runs correctly using a Mock pipeline.
    """
    
    with patch("SwiftLyrics_Analytics.nn_analysis.pipeline") as mock_pipeline:
        
        mock_classifier = MagicMock()
        mock_classifier.return_value = {"labels": ["positive"]} 
        mock_pipeline.return_value = mock_classifier

        result_df = nn_analysis.classify(sample_dataframe, album_id="1989")

        assert result_df is not None
        assert not result_df.empty

        assert "sentiment" in result_df.columns
        assert "theme" in result_df.columns

        assert result_df.iloc[0]["album_id"] == "1989"
        
        assert result_df.iloc[0]["sentiment"] == "positive"