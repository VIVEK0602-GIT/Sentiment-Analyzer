from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

def analyze_sentiment_vader(text):
    """
    Analyze the sentiment of a text using VADER.
    Returns 'positive', 'negative', or 'neutral'.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def analyze_text_corpus(texts):
    """
    Analyze a list of texts and return sentiment labels.
    """
    return [analyze_sentiment_vader(text) for text in texts]

def vader_score_dataset(df, text_column='review'):
    """
    Score an entire dataset with VADER and add sentiment predictions
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Apply VADER analysis
    result_df['vader_sentiment'] = analyze_text_corpus(result_df[text_column])
    
    # Calculate agreement with ground truth if available
    if 'sentiment' in result_df.columns:
        result_df['agreement'] = result_df['sentiment'] == result_df['vader_sentiment']
        agreement_rate = result_df['agreement'].mean()
        print(f"VADER agreement with labeled data: {agreement_rate:.4f}")
    
    return result_df

def get_detailed_scores(texts):
    """
    Get detailed VADER scores (pos, neg, neu, compound) for a list of texts
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    
    for text in texts:
        scores = analyzer.polarity_scores(text)
        results.append(scores)
    
    return pd.DataFrame(results) 