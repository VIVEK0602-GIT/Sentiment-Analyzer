from src.vader import analyze_sentiment_vader, get_detailed_scores
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "This movie was absolutely fantastic! I loved every minute of it."
    
    sentiment = analyze_sentiment_vader(text)
    scores = get_detailed_scores([text]).iloc[0].to_dict()
    
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}") 