# Sentiment Analysis Pro

A modern, modular, and high-performance sentiment analysis project for movie reviews. Supports both VADER (lexicon-based) and ML (Logistic Regression + TF-IDF) models, with a beautiful Streamlit dashboard for single and batch analysis.

---

## 📑 Index
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Single Review Analysis](#single-review-analysis)
  - [Batch Analysis](#batch-analysis)
  - [VADER vs ML Model](#vader-vs-ml-model)
- [Training & Retraining](#training--retraining)
- [File Structure](#file-structure)
- [Extending the Project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project provides a robust sentiment analysis pipeline for movie reviews, featuring:
- Fast VADER analysis for quick results
- Accurate ML model (Logistic Regression + TF-IDF) for production use
- Interactive Streamlit dashboard for both single and batch review analysis
- Easy retraining and extensibility

## Features
- 📊 **Streamlit dashboard**: Modern, interactive, and easy to use
- ⚡ **VADER**: Fast, lexicon-based sentiment analysis
- 🤖 **ML Model**: Logistic Regression with TF-IDF for higher accuracy
- 🗂️ **Batch & Single Review**: Analyze one or thousands of reviews
- 📈 **Interactive Charts**: Sentiment gauge, breakdown, word impact
- ⬇️ **Download Results**: Export batch results as CSV
- 🛠️ **Easy retraining**: Use your own data or the full IMDB dataset

## Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. **Open in your browser**: [http://localhost:8501](http://localhost:8501) (or as shown in your terminal)

## Usage
### Single Review Analysis
- Enter your review in the text box and click "Analyze".
- Choose between VADER (fast) and ML Model (accurate) in the sidebar.

### Batch Analysis
- Upload a CSV file with a `review` column (and optional `sentiment` column).
- View full results, download as CSV, and see summary stats.

### VADER vs ML Model
- **VADER**: Fast, no training needed, good for social media/short texts.
- **ML Model**: Trained on 50,000 IMDB reviews, higher accuracy for movie reviews.

## Training & Retraining
- To retrain the ML model on new data, use:
  ```bash
  python -c "from src.ml_model import train_model; train_model('data/your_data.csv', 'models/logreg_model.joblib', 'models/tfidf_vectorizer.joblib')"
  ```
- To download and prepare the full IMDB dataset:
  ```bash
  python src/download_and_prepare_imdb.py
  ```

## File Structure
```
app/
  streamlit_app.py         # Main dashboard app
src/
  preprocess.py            # Text cleaning and preprocessing
  sentiment.py             # VADER and ML model interfaces
  ml_model.py              # ML model training/loading
  visualization.py         # Plotly chart utilities
  download_and_prepare_imdb.py # IMDB dataset downloader
models/
  logreg_model.joblib      # Trained ML model
  tfidf_vectorizer.joblib  # Trained vectorizer
requirements.txt           # All dependencies
README.md                  # This file
```

## Extending the Project
- Add new models (SVM, Random Forest, etc.) in `src/ml_model.py`
- Add new visualizations in `src/visualization.py`
- Use your own datasets for training

## Troubleshooting
- **ModuleNotFoundError: No module named 'src'**: Make sure `sys.path` is set at the top of `app/streamlit_app.py` (already handled in this repo).
- **Batch results only show a few rows**: Now fixed to show all results.
- **Slow training**: Use a smaller dataset or run on a machine with more RAM/CPU.

## Contributing
Pull requests and suggestions are welcome! Please open an issue or PR.

## License
MIT License. See [LICENSE](LICENSE) for details.