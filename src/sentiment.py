import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return {**scores, 'sentiment': sentiment}

def analyze_vader_batch(texts):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        results.append({**scores, 'sentiment': sentiment, 'text': text})
    return pd.DataFrame(results)

def analyze_ml(text, model, vectorizer, preprocessor):
    X = vectorizer.transform([preprocessor(text)])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {'sentiment': pred, 'proba': proba}

def analyze_ml_batch(texts, model, vectorizer, preprocessor):
    X = vectorizer.transform([preprocessor(t) for t in texts])
    preds = model.predict(X)
    probas = model.predict_proba(X)
    return pd.DataFrame({
        'text': texts,
        'sentiment': preds,
        'proba_positive': probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
    }) 