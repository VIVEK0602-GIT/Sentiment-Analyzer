import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import os
from src.sentiment import analyze_vader, analyze_vader_batch, analyze_ml, analyze_ml_batch
from src.ml_model import load_model
from src.visualization import plot_gauge, plot_breakdown, plot_word_impact
from src.preprocess import get_preprocessor

st.set_page_config(page_title="Sentiment Analysis Pro", layout="wide")

st.sidebar.title("Sentiment Analysis Pro")
mode = st.sidebar.radio("Mode", ["Single Review", "Batch Upload"])
engine = st.sidebar.radio("Engine", ["VADER (Fast)", "ML Model (Accurate)"])

# Load ML model if needed
ml_model, ml_vectorizer, ml_preprocessor = None, None, None
if engine == "ML Model (Accurate)":
    model_path = "models/logreg_model.joblib"
    vectorizer_path = "models/tfidf_vectorizer.joblib"
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        ml_model, ml_vectorizer, ml_preprocessor = load_model(model_path, vectorizer_path)
    else:
        st.warning("ML model not found. Please train the model first.")

if mode == "Single Review":
    text = st.text_area("Enter your review:")
    if st.button("Analyze"):
        if engine == "VADER (Fast)":
            result = analyze_vader(text)
            st.plotly_chart(plot_gauge(result['compound']))
            st.plotly_chart(plot_breakdown(result))
            # Simple word impact (for demo)
            word_impacts = [(w, 0.7, 'positive') for w in text.split() if w.lower() in ["great","excellent","amazing","good","love","best"]]
            st.plotly_chart(plot_word_impact(word_impacts))
            st.write(f"**Sentiment:** {result['sentiment'].capitalize()}")
        elif engine == "ML Model (Accurate)" and ml_model:
            result = analyze_ml(text, ml_model, ml_vectorizer, ml_preprocessor)
            st.write(f"**Sentiment:** {result['sentiment'].capitalize()}")
            st.write(f"**Probabilities:** {result['proba']}")
else:
    file = st.file_uploader("Upload CSV (columns: review,sentiment)", type="csv")
    if file:
        df = pd.read_csv(file)
        # Remove empty reviews
        original_count = len(df)
        df = df[df['review'].astype(str).str.strip() != '']
        processed_count = len(df)
        skipped_count = original_count - processed_count
        if engine == "VADER (Fast)":
            results = analyze_vader_batch(df['review'].astype(str))
            st.dataframe(results)  # Show all results
            st.plotly_chart(plot_breakdown(results))
            st.download_button("Download Results", results.to_csv(index=False), "results.csv")
        elif engine == "ML Model (Accurate)" and ml_model:
            results = analyze_ml_batch(df['review'].astype(str), ml_model, ml_vectorizer, ml_preprocessor)
            st.dataframe(results)  # Show all results
            st.download_button("Download Results", results.to_csv(index=False), "results.csv")
        st.info(f"Processed {processed_count} reviews. Skipped {skipped_count} empty or invalid rows.")

st.markdown("---")
st.markdown("Sentiment analysis powered by VADER and Logistic Regression (TF-IDF)")
st.markdown("Created by NLP Project Team") 