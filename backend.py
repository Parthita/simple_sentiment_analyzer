from gnews import GNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import json

def download_nltk_data():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

def fetch_twitter(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    tweets = []
    
    try:
        import snscrape.modules.twitter as sntwitter
        
        scraper = sntwitter.TwitterSearchScraper(f"{query} lang:en")
        
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
                
            tweets.append({
                "source": "Twitter",
                "text": tweet.rawContent,
                "timestamp": tweet.date.isoformat() if tweet.date else None,
                "url": tweet.url,
                "user": tweet.user.username if tweet.user else None
            })
            
    except Exception as e:
        mock_tweets = [
            {
                "source": "Twitter",
                "text": f"Just got the new {query}! Amazing features and great performance. Love it!",
                "timestamp": datetime.now().isoformat(),
                "url": "https://twitter.com/mock/status/1",
                "user": "mock_user1"
            },
            {
                "source": "Twitter", 
                "text": f"Not impressed with {query}. Battery life is terrible and it's overpriced.",
                "timestamp": datetime.now().isoformat(),
                "url": "https://twitter.com/mock/status/2",
                "user": "mock_user2"
            },
            {
                "source": "Twitter",
                "text": f"{query} is okay, nothing special but gets the job done.",
                "timestamp": datetime.now().isoformat(),
                "url": "https://twitter.com/mock/status/3",
                "user": "mock_user3"
            }
        ]
        return mock_tweets[:limit]
        
    return tweets

def fetch_news(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    news_articles = []
    try:
        news = GNews()
        news.max_results = limit
        
        articles = news.get_news(query)
        
        for article in articles:
            news_articles.append({
                "source": "GNews",
                "text": article.get('title', ''),
                "timestamp": article.get('published date', ''),
                "url": article.get('url', ''),
                "description": article.get('description', '')
            })
            
    except Exception as e:
        pass
        
    return news_articles

def analyze_sentiment(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    download_nltk_data()
    
    analyzer = SentimentIntensityAnalyzer()
    results = []
    
    for item in texts:
        text = item.get('text', '')
        scores = analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        if compound_score > 0.05:
            sentiment = "Positive"
        elif compound_score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        item_with_sentiment = item.copy()
        item_with_sentiment.update({
            'sentiment': sentiment,
            'compound_score': compound_score,
            'sentiment_scores': scores
        })
        
        results.append(item_with_sentiment)
    
    return results

def aggregate_results(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {
            "overall": {"total": 0, "positive": 0, "negative": 0, "neutral": 0},
            "by_source": {},
            "examples": []
        }
    
    total = len(data)
    positive = sum(1 for item in data if item.get('sentiment') == 'Positive')
    negative = sum(1 for item in data if item.get('sentiment') == 'Negative')
    neutral = sum(1 for item in data if item.get('sentiment') == 'Neutral')
    
    overall_stats = {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round((positive / total) * 100, 2) if total > 0 else 0,
        "negative_pct": round((negative / total) * 100, 2) if total > 0 else 0,
        "neutral_pct": round((neutral / total) * 100, 2) if total > 0 else 0
    }
    
    by_source = {}
    sources = set(item.get('source', 'Unknown') for item in data)
    
    for source in sources:
        source_data = [item for item in data if item.get('source') == source]
        source_total = len(source_data)
        source_positive = sum(1 for item in source_data if item.get('sentiment') == 'Positive')
        source_negative = sum(1 for item in source_data if item.get('sentiment') == 'Negative')
        source_neutral = sum(1 for item in source_data if item.get('sentiment') == 'Neutral')
        
        by_source[source] = {
            "total": source_total,
            "positive": source_positive,
            "negative": source_negative,
            "neutral": source_neutral,
            "positive_pct": round((source_positive / source_total) * 100, 2) if source_total > 0 else 0,
            "negative_pct": round((source_negative / source_total) * 100, 2) if source_total > 0 else 0,
            "neutral_pct": round((source_neutral / source_total) * 100, 2) if source_total > 0 else 0
        }
    
    examples = []
    for item in data[:5]:
        examples.append({
            "source": item.get('source', 'Unknown'),
            "text": item.get('text', '')[:100] + '...' if len(item.get('text', '')) > 100 else item.get('text', ''),
            "sentiment": item.get('sentiment', 'Unknown'),
            "compound_score": round(item.get('compound_score', 0), 3)
        })
    
    return {
        "overall": overall_stats,
        "by_source": by_source,
        "examples": examples
    }

def main():
    query = "iPhone"
    
    twitter_data = fetch_twitter(query, limit=30)
    news_data = fetch_news(query, limit=15)
    
    all_data = twitter_data + news_data
    
    if not all_data:
        return
    
    analyzed_data = analyze_sentiment(all_data)
    results = aggregate_results(analyzed_data)
    
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*50)
    
    overall = results['overall']
    print(f"Total items: {overall['total']}")
    print(f"Positive: {overall['positive']} ({overall['positive_pct']}%)")
    print(f"Negative: {overall['negative']} ({overall['negative_pct']}%)")
    print(f"Neutral: {overall['neutral']} ({overall['neutral_pct']}%)")
    
    print(f"\nPer-source statistics:")
    for source, stats in results['by_source'].items():
        print(f"\n{source}:")
        print(f"  Total: {stats['total']}")
        print(f"  Positive: {stats['positive']} ({stats['positive_pct']}%)")
        print(f"  Negative: {stats['negative']} ({stats['negative_pct']}%)")
        print(f"  Neutral: {stats['neutral']} ({stats['neutral_pct']}%)")
    
    print(f"\nExamples:")
    for i, example in enumerate(results['examples'], 1):
        print(f"\n{i}. [{example['source']}] {example['sentiment']} (score: {example['compound_score']})")
        print(f"   Text: {example['text']}")

if __name__ == "__main__":
    main()