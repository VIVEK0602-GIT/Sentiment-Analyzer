import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess import get_preprocessor

def train_model(data_path, model_path, vectorizer_path):
    df = pd.read_csv(data_path)
    preprocessor = get_preprocessor()
    X = df['review'].astype(str).apply(preprocessor)
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return acc

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    preprocessor = get_preprocessor()
    return model, vectorizer, preprocessor 