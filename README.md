# News Sentiment Analysis Dashboard

An interactive Streamlit app that fetches the latest news headlines and analyzes their sentiment using VADER. Explore topic sentiment at a glance with charts, word clouds, and downloadable results.

## Overview and Screenshots

The dashboard lets you:

- Filter news by keyword/topic
- Run sentiment analysis (positive/neutral/negative, compound)
- Visualize distributions and trends
- Inspect headline-level scores and export results

Screenshots:

![Dashboard Home](pictures/20250908001435)
![Analysis Results](pictures/20250908001509)
![Word Cloud](pictures/20250908001516)
![Time Series](pictures/20250908001522)
![Details Table](pictures/20250908001525)

## Data Source and Rationale

- **Source**: GNews (community Python wrapper around Google News)
- **Why GNews?**
  - Broad, near real-time news coverage across many outlets
  - Simple query interface suitable for rapid prototyping and demos
  - Lightweight dependency that returns headline text and metadata sufficient for sentiment analysis

## Sentiment Model and Justification

- **Model**: VADER (`nltk.sentiment.vader.SentimentIntensityAnalyzer`)
- **Why VADER?**
  - Designed for short, informal text like headlines and social posts
  - Rule- and lexicon-based: works well out-of-the-box without task-specific training
  - Robust handling of punctuation, capitalization, negations, and degree modifiers

## Local Setup and Execution

### Prerequisites

- Python 3.9+
- pip

### 1) Clone the repository

```bash
git clone <this-repo-url>
cd gdgd_sentiment_analyzer
```

### 2) (Optional) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

Either install from the list below or create a `requirements.txt` with these packages and run `pip install -r requirements.txt`:

```bash
pip install \
  streamlit>=1.28.0 \
  plotly>=5.15.0 \
  matplotlib>=3.7.0 \
  wordcloud>=1.9.0 \
  nltk>=3.8.1 \
  pandas>=1.5.0 \
  gnews>=0.4.2
```

On first run, NLTK may download the VADER lexicon automatically. If needed, you can pre-download it:

```python
import nltk
nltk.download('vader_lexicon')
```

### 4) Run the app

The Streamlit entry points found in this repo are `app.py` and `backend.py`. To start the UI:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser. Enter a topic, adjust any parameters, and click "Analyze Sentiment".

## Features

- **News Data**: Fetches news headlines from GNews
- **Sentiment Analysis**: Uses VADER SentimentIntensityAnalyzer
- **Visualizations**: Interactive charts and word clouds
- **Data Export**: Download results as CSV or JSON

## Deployment

This app is ready for deployment on Streamlit Community Cloud.

## License

MIT License