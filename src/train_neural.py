import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import local modules
from neural_models import NeuralSentimentClassifier
from preprocess import preprocess_dataframe, generate_wordcloud, plot_word_frequencies_by_sentiment
from utils import load_data, save_data, plot_class_distribution
from evaluate import evaluate_model, save_evaluation_results

def load_and_preprocess_data(data_path, use_existing=False):
    """
    Load and preprocess the dataset
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    use_existing : bool
        Whether to use existing preprocessed data
        
    Returns:
    --------
    dict
        Dictionary containing the preprocessed data
    """
    # Check if preprocessed data exists and should be used
    processed_dir = os.path.dirname(data_path)
    processed_path = os.path.join(processed_dir, 'preprocessed_neural.csv')
    
    if use_existing and os.path.exists(processed_path):
        print(f"Loading preprocessed data from {processed_path}")
        processed_df = pd.read_csv(processed_path)
        return {
            'processed_df': processed_df,
            'X': processed_df['processed_text'],
            'y': processed_df['sentiment']
        }
    
    # Load the original data
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    print(f"Loaded {len(df)} records")
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Preprocess the data with additional text features
    print("\nPreprocessing data...")
    processed_df = preprocess_dataframe(
        df, 
        text_column='review', 
        remove_stops=True, 
        lemmatize=True,
        stem=False,
        handle_neg=True,
        use_pos_tagging=True,
        extract_features=True
    )
    
    # Save the preprocessed data
    save_data(processed_df, processed_path)
    print(f"Saved preprocessed data to {processed_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Generate word clouds for each sentiment
    for sentiment in df['sentiment'].unique():
        sentiment_texts = processed_df[processed_df['sentiment'] == sentiment]['processed_text']
        generate_wordcloud(
            sentiment_texts, 
            sentiment=sentiment,
            save_path=f'results/visualizations/wordcloud_{sentiment}.png'
        )
    
    # Plot word frequencies by sentiment
    plot_word_frequencies_by_sentiment(
        processed_df,
        save_path='results/visualizations/word_frequencies_by_sentiment.png'
    )
    
    return {
        'processed_df': processed_df,
        'X': processed_df['processed_text'],
        'y': processed_df['sentiment']
    }

def train_neural_models(data_dict, model_types=None, epochs=10, batch_size=32):
    """
    Train neural network models for sentiment analysis
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the preprocessed data
    model_types : list
        List of model types to train
    epochs : int
        Number of epochs to train
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    dict
        Dictionary containing trained models and evaluation results
    """
    if model_types is None:
        model_types = ['cnn', 'lstm', 'bilstm']
    
    # Extract data
    X = data_dict['X']
    y = data_dict['y']
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train and evaluate models
    results = {}
    metrics_list = []
    
    for model_type in model_types:
        print(f"\n--- Training {model_type.upper()} model ---")
        
        # Create classifier
        classifier = NeuralSentimentClassifier(model_type=model_type)
        
        # Preprocess data for neural network
        X_train_seq, y_train_enc = classifier.preprocess_data(X_train, y_train, train=True)
        X_val_seq, y_val_enc = classifier.preprocess_data(X_val, y_val)
        X_test_seq, y_test_enc = classifier.preprocess_data(X_test, y_test)
        
        # Build and train model
        num_classes = len(np.unique(y_train_enc))
        classifier.build_model(num_classes)
        
        # Train model
        history = classifier.train(
            X_train_seq, 
            y_train_enc, 
            X_val=X_val_seq, 
            y_val=y_val_enc,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Plot training history
        classifier.plot_training_history(
            save_path=f'results/visualizations/{model_type}_training_history.png'
        )
        
        # Evaluate model
        metrics = classifier.evaluate(X_test_seq, y_test_enc)
        print(f"\n{model_type.upper()} evaluation:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Make predictions for comprehensive evaluation
        y_pred = classifier.predict(X_test)
        
        # Perform comprehensive evaluation
        eval_metrics = evaluate_model(
            classifier, 
            X_test_seq, 
            y_test_enc, 
            model_name=f'Neural {model_type.upper()}'
        )
        
        # Add trained model and class names to metrics
        eval_metrics['model'] = classifier
        eval_metrics['class_names'] = classifier.label_encoder.classes_
        
        # Add to results
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics,
            'history': history,
            'eval_metrics': eval_metrics
        }
        
        # Add to metrics list for comparison
        metrics_list.append(eval_metrics)
    
    # Save comprehensive evaluation results
    print("\nSaving evaluation results...")
    save_paths = save_evaluation_results(metrics_list, prefix='neural')
    
    return {
        'results': results,
        'metrics_list': metrics_list,
        'save_paths': save_paths,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        }
    }

def main():
    """
    Main function to train neural network models for sentiment analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network models for sentiment analysis')
    parser.add_argument('--data', type=str, default='data/larger_dataset/train.csv', help='Path to the dataset')
    parser.add_argument('--models', type=str, nargs='+', default=['cnn', 'lstm', 'bilstm'], help='Model types to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use-existing', action='store_true', help='Use existing preprocessed data')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    data_dict = load_and_preprocess_data(args.data, use_existing=args.use_existing)
    
    # Train models
    results = train_neural_models(
        data_dict,
        model_types=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\nNeural network models training completed successfully!")
    print(f"Results saved in the 'results' directory.")

if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main() 