import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
import numpy as np

from backend import fetch_news, analyze_sentiment, aggregate_results

st.set_page_config(
    page_title="Multi-Source Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'aggregated_results' not in st.session_state:
    st.session_state.aggregated_results = None

def generate_wordcloud(texts: List[str], title: str, color_scheme: str = 'viridis') -> None:
    if not texts:
        st.warning(f"No texts available for {title} word cloud")
        return
    
    combined_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=color_scheme,
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    st.pyplot(fig)

def create_timeline_chart(data: List[Dict[str, Any]]) -> go.Figure:
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if df.empty:
        return go.Figure()
    
    daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    daily_percentages = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    
    colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in daily_percentages.columns:
            fig.add_trace(go.Scatter(
                x=daily_percentages.index,
                y=daily_percentages[sentiment],
                mode='lines+markers',
                name=sentiment,
                line=dict(color=colors[sentiment], width=3),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title="Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">Multi-Source Sentiment Dashboard</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Analysis Controls")
        
        query = st.text_input(
            "Search Query",
            value="iPhone",
            help="Enter a topic or keyword to analyze sentiment"
        )
        
        st.subheader("Parameters")
        news_limit = st.slider("News Articles", 5, 50, 20)
        
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        
        st.subheader("Optional Features")
        show_timeline = st.checkbox("Show Timeline", value=True)
        show_wordcloud = st.checkbox("Show Word Clouds", value=True)
    
    if analyze_btn and query:
        with st.spinner("Analyzing sentiment... This may take a moment."):
            st.info("Fetching news data...")
            news_data = fetch_news(query, news_limit)
            
            if not news_data:
                st.error("No data found. Please try a different query.")
                return
            
            st.info("Analyzing sentiment...")
            analyzed_data = analyze_sentiment(news_data)
            
            st.info("Aggregating results...")
            aggregated_results = aggregate_results(analyzed_data)
            
            st.session_state.analysis_data = analyzed_data
            st.session_state.aggregated_results = aggregated_results
    
    if st.session_state.analysis_data and st.session_state.aggregated_results:
        data = st.session_state.analysis_data
        results = st.session_state.aggregated_results
        
        st.header("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Texts",
                value=results['overall']['total'],
                delta=None
            )
        
        with col2:
            positive_pct = results['overall']['positive_pct']
            st.metric(
                label="Positive",
                value=f"{positive_pct:.1f}%",
                delta=f"+{results['overall']['positive']} texts"
            )
        
        with col3:
            negative_pct = results['overall']['negative_pct']
            st.metric(
                label="Negative",
                value=f"{negative_pct:.1f}%",
                delta=f"+{results['overall']['negative']} texts"
            )
        
        with col4:
            neutral_pct = results['overall']['neutral_pct']
            st.metric(
                label="Neutral",
                value=f"{neutral_pct:.1f}%",
                delta=f"+{results['overall']['neutral']} texts"
            )
        
        st.header("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = {
                'Positive': results['overall']['positive'],
                'Negative': results['overall']['negative'],
                'Neutral': results['overall']['neutral']
            }
            
            fig_pie = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            source_data = []
            for source, stats in results['by_source'].items():
                source_data.extend([
                    {'Source': source, 'Sentiment': 'Positive', 'Count': stats['positive']},
                    {'Source': source, 'Sentiment': 'Negative', 'Count': stats['negative']},
                    {'Source': source, 'Sentiment': 'Neutral', 'Count': stats['neutral']}
                ])
            
            df_source = pd.DataFrame(source_data)
            fig_bar = px.bar(
                df_source,
                x='Source',
                y='Count',
                color='Sentiment',
                title="Sentiment by Source",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        if show_timeline:
            st.subheader("Sentiment Timeline")
            timeline_fig = create_timeline_chart(data)
            if timeline_fig.data:
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("Timeline data not available (insufficient timestamp data)")
        
        if show_wordcloud:
            st.subheader("Word Clouds")
            
            positive_texts = [item['text'] for item in data if item.get('sentiment') == 'Positive']
            negative_texts = [item['text'] for item in data if item.get('sentiment') == 'Negative']
            
            col1, col2 = st.columns(2)
            
            with col1:
                generate_wordcloud(positive_texts, "Positive Sentiment", 'Greens')
            
            with col2:
                generate_wordcloud(negative_texts, "Negative Sentiment", 'Reds')
        
        st.subheader("Recent Mentions")
        
        display_data = []
        for item in data[:20]:
            display_data.append({
                'Text': item['text'][:100] + '...' if len(item['text']) > 100 else item['text'],
                'Source': item['source'],
                'Sentiment': item.get('sentiment', 'Unknown'),
                'Score': round(item.get('compound_score', 0), 3),
                'Timestamp': item.get('timestamp', 'N/A')
            })
        
        df_display = pd.DataFrame(display_data)
        
        def color_sentiment(val):
            if val == 'Positive':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Negative':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Neutral':
                return 'background-color: #e2e3e5; color: #383d41'
            return ''
        
        styled_df = df_display.style.map(color_sentiment, subset=['Sentiment'])
        st.dataframe(styled_df, width='stretch')
        
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"sentiment_analysis_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'data': data
            }
            st.download_button(
                label="Download as JSON",
                data=pd.Series([json_data]).to_json(orient='records', indent=2),
                file_name=f"sentiment_analysis_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.markdown("""
        ## Welcome to the Multi-Source Sentiment Dashboard!
        
        This dashboard helps you analyze sentiment across multiple sources:
        
        - **News**: Media coverage sentiment analysis
        - **Analytics**: Comprehensive sentiment breakdown
        - **Word Clouds**: Visual text analysis
        
        ### Getting Started
        
        1. Enter a search query in the sidebar
        2. Adjust the parameters (News articles)
        3. Click "Analyze Sentiment" to start
        4. Explore the results in the dashboard
        
        ### Tips
        
        - Try different queries to see how sentiment varies
        - Use the optional features to get deeper insights
        - Export your data for further analysis
        """)
        
        example_queries = [
            "iPhone", "Tesla", "Bitcoin", "Climate Change", "Artificial Intelligence",
            "Electric Vehicles", "Renewable Energy", "Space Exploration", "COVID-19"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(example_queries):
            with cols[i % 3]:
                if st.button(f"{query}", key=f"example_{i}"):
                    st.session_state.example_query = query
                    st.rerun()

if __name__ == "__main__":
    main()