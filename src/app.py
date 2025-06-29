from flask import Flask, request, jsonify, render_template, url_for
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
import traceback

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.preprocess import preprocess_text
from src.vader import analyze_sentiment_vader, get_detailed_scores
try:
    from src.neural_models import NeuralSentimentClassifier
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__, template_folder='../web/templates', static_folder='../web/static')

# Load traditional models
def load_traditional_models():
    models = {}
    
    try:
        # Load logistic regression model
        lr_model = load('../models/logistic_regression.pkl')
        lr_vectorizer = load('../models/logistic_regression_vectorizer.pkl')
        models['logistic_regression'] = {
            'model': lr_model,
            'vectorizer': lr_vectorizer
        }
        
        # Load SVM model
        svm_model = load('../models/svm.pkl')
        svm_vectorizer = load('../models/svm_vectorizer.pkl')
        models['svm'] = {
            'model': svm_model,
            'vectorizer': svm_vectorizer
        }
        
        return models, True
    except Exception as e:
        print(f"Error loading traditional models: {e}")
        traceback.print_exc()
        return {}, False

# Load neural models
def load_neural_models():
    models = {}
    
    if not NEURAL_MODELS_AVAILABLE:
        return models, False
    
    try:
        # Check which models are available
        model_types = ['cnn', 'lstm', 'bilstm']
        for model_type in model_types:
            model_path = f'../models/{model_type}_model.h5'
            if os.path.exists(model_path):
                # Load the model
                classifier = NeuralSentimentClassifier.load_model(model_type=model_type, model_path='../models')
                models[model_type] = classifier
        
        if models:
            return models, True
        else:
            return {}, False
    except Exception as e:
        print(f"Error loading neural models: {e}")
        traceback.print_exc()
        return {}, False

# Load models
TRADITIONAL_MODELS, TRADITIONAL_MODELS_AVAILABLE = load_traditional_models()
NEURAL_MODELS, NEURAL_MODELS_AVAILABLE = load_neural_models()

# Define routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                          traditional_models_available=TRADITIONAL_MODELS_AVAILABLE,
                          neural_models_available=NEURAL_MODELS_AVAILABLE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the text sentiment"""
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'vader')
    
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    # Preprocess the text
    processed_text = preprocess_text(text, remove_stops=True, lemmatize=True)
    
    results = {}
    detailed_results = {}
    
    try:
        # VADER analysis (always included)
        vader_sentiment = analyze_sentiment_vader(text)
        vader_scores = get_detailed_scores([text]).iloc[0].to_dict()
        
        results['vader'] = vader_sentiment
        detailed_results['vader'] = {
            'sentiment': vader_sentiment,
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
        
        # Traditional ML models
        if model_type in ['logistic_regression', 'svm'] and TRADITIONAL_MODELS_AVAILABLE:
            model_info = TRADITIONAL_MODELS[model_type]
            model = model_info['model']
            vectorizer = model_info['vectorizer']
            
            # Vectorize the text
            X = vectorizer.transform([processed_text])
            
            # Predict
            prediction = model.predict(X)[0]
            
            results[model_type] = prediction
            detailed_results[model_type] = {
                'sentiment': prediction
            }
            
            # Add probabilities if available
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)[0]
                classes = model.classes_
                
                proba_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
                detailed_results[model_type]['probabilities'] = proba_dict
        
        # Neural models
        elif model_type in NEURAL_MODELS and NEURAL_MODELS_AVAILABLE:
            classifier = NEURAL_MODELS[model_type]
            
            # Predict
            prediction = classifier.predict([text])[0]
            
            results[model_type] = prediction
            detailed_results[model_type] = {
                'sentiment': prediction
            }
        
        return jsonify({
            'text': text,
            'processed_text': processed_text,
            'results': results,
            'detailed_results': detailed_results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare all available models"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    # Preprocess the text
    processed_text = preprocess_text(text, remove_stops=True, lemmatize=True)
    
    results = {}
    detailed_results = {}
    
    try:
        # VADER analysis
        vader_sentiment = analyze_sentiment_vader(text)
        vader_scores = get_detailed_scores([text]).iloc[0].to_dict()
        
        results['vader'] = vader_sentiment
        detailed_results['vader'] = {
            'sentiment': vader_sentiment,
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
        
        # Traditional ML models
        if TRADITIONAL_MODELS_AVAILABLE:
            for model_name, model_info in TRADITIONAL_MODELS.items():
                model = model_info['model']
                vectorizer = model_info['vectorizer']
                
                # Vectorize the text
                X = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(X)[0]
                
                results[model_name] = prediction
                detailed_results[model_name] = {
                    'sentiment': prediction
                }
                
                # Add probabilities if available
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)[0]
                    classes = model.classes_
                    
                    proba_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
                    detailed_results[model_name]['probabilities'] = proba_dict
        
        # Neural models
        if NEURAL_MODELS_AVAILABLE:
            for model_name, classifier in NEURAL_MODELS.items():
                # Predict
                prediction = classifier.predict([text])[0]
                
                results[model_name] = prediction
                detailed_results[model_name] = {
                    'sentiment': prediction
                }
        
        return jsonify({
            'text': text,
            'processed_text': processed_text,
            'results': results,
            'detailed_results': detailed_results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/models')
def get_models():
    """Get the list of available models"""
    models = []
    
    # Add VADER
    models.append({
        'id': 'vader',
        'name': 'VADER',
        'type': 'lexicon-based',
        'description': 'Valence Aware Dictionary and sEntiment Reasoner - A rule-based sentiment analysis tool'
    })
    
    # Add traditional ML models
    if TRADITIONAL_MODELS_AVAILABLE:
        models.append({
            'id': 'logistic_regression',
            'name': 'Logistic Regression',
            'type': 'traditional-ml',
            'description': 'A linear model for classification that uses a logistic function'
        })
        
        models.append({
            'id': 'svm',
            'name': 'Support Vector Machine',
            'type': 'traditional-ml',
            'description': 'A model that finds the hyperplane that best divides the classes'
        })
    
    # Add neural models
    if NEURAL_MODELS_AVAILABLE:
        for model_name in NEURAL_MODELS:
            if model_name == 'cnn':
                models.append({
                    'id': 'cnn',
                    'name': 'Convolutional Neural Network',
                    'type': 'neural',
                    'description': 'A deep learning model that uses convolutional layers for text classification'
                })
            elif model_name == 'lstm':
                models.append({
                    'id': 'lstm',
                    'name': 'Long Short-Term Memory',
                    'type': 'neural',
                    'description': 'A recurrent neural network architecture for sequence modeling'
                })
            elif model_name == 'bilstm':
                models.append({
                    'id': 'bilstm',
                    'name': 'Bidirectional LSTM',
                    'type': 'neural',
                    'description': 'A bidirectional LSTM that processes the sequence in both directions'
                })
    
    return jsonify({
        'models': models
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'traditional_models_available': TRADITIONAL_MODELS_AVAILABLE,
        'neural_models_available': NEURAL_MODELS_AVAILABLE
    })

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('../web/templates', exist_ok=True)
    os.makedirs('../web/static/css', exist_ok=True)
    os.makedirs('../web/static/js', exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 