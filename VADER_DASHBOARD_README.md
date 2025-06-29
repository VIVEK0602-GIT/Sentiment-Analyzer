# Sentiment Analysis Dashboard

A modern, interactive tool for sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).

## Features

- Analyze individual text inputs for sentiment
- Real-time sentiment analysis with visual feedback
- Detailed sentiment breakdown with positive, negative, and neutral scores
- Easy-to-use web interface with Streamlit
- No TensorFlow or complex dependencies required

## Quick Start

### Windows

Simply run the included batch file:

```
run_streamlit_dashboard.bat
```

This will:
1. Check for Python installation
2. Install all required dependencies
3. Launch the Streamlit dashboard

### Any Platform

Run the Streamlit app directly:

```bash
# Install required dependencies
pip install streamlit validators "tzlocal<6" "importlib-metadata<7" "packaging<24" "protobuf<5" "rich<14" "tenacity<9" numpy==1.22.3 vaderSentiment

# Run the app
streamlit run sentiment_streamlit_app.py
```

## Using the Dashboard

1. **Enter Text**: Type any text you want to analyze in the text input area
2. **Use Sample**: Try one of the provided sample texts from the sidebar
3. **Analyze**: Click the "Analyze Sentiment" button to process the text
4. **View Results**: See a detailed breakdown of sentiment scores
5. **Explanation**: Read an explanation of what the sentiment means

## Dependencies

The application uses the following key dependencies:

- Python 3.6+
- Streamlit (for the web interface)
- VADER Sentiment Analysis (from NLTK)
- Various support libraries (automatically installed by the batch file)

## Examples

**Positive text**: "This product exceeded all my expectations! The quality is outstanding."
- Result: Positive sentiment (compound score: 0.8316)

**Negative text**: "Terrible experience. The customer service was awful and the product didn't work."
- Result: Negative sentiment (compound score: -0.8225)

**Neutral text**: "It's an okay product. Not amazing but gets the job done."
- Result: Neutral sentiment (compound score: 0.0258)

## Troubleshooting

If you encounter any errors:

1. **Missing Dependencies**: Run the batch file which will install all required packages
2. **Python Version**: Ensure you're using Python 3.6 or newer
3. **Streamlit Issues**: If Streamlit fails to start, try running `pip install --upgrade streamlit==1.29.0`

## About VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media and works well on texts from other domains.

VADER is particularly suitable when:
- You need quick, light-weight sentiment analysis
- You don't have labeled data for training
- You want to analyze sentiment without deep learning models 