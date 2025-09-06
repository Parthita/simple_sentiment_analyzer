# Multi-Source Sentiment Analysis Dashboard

A sentiment analysis dashboard that analyzes sentiment across Twitter and news sources using VADER sentiment analysis.

## Features

- **Twitter Data**: Fetches recent tweets (with mock data fallback)
- **News Data**: Fetches news headlines from GNews
- **Sentiment Analysis**: Uses VADER SentimentIntensityAnalyzer
- **Visualizations**: Interactive charts and word clouds
- **Data Export**: Download results as CSV or JSON

## Usage

1. Enter a search query in the sidebar
2. Adjust parameters (Twitter posts, News articles)
3. Click "Analyze Sentiment" to start
4. Explore results and export data

## Dependencies

- streamlit>=1.28.0
- plotly>=5.15.0
- matplotlib>=3.7.0
- wordcloud>=1.9.0
- nltk>=3.8.1
- pandas>=1.5.0
- gnews>=0.4.2
- snscrape>=0.6.2

## Deployment

This app is ready for deployment on Streamlit Community Cloud.

## License

MIT License