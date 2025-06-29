import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sentiment_analysis")

# Import local modules
try:
    from utils import load_data, save_data, plot_class_distribution, display_sample_records, plot_word_frequencies
    from preprocess import preprocess_dataframe, preprocess_text
    from train import train_models, load_model, load_vectorizer, build_feature_matrix
    from evaluate import evaluate_model, plot_confusion_matrix, save_metrics_to_csv, plot_metrics_comparison, plot_roc_curve, analyze_misclassifications, plot_feature_importance, create_interactive_dashboard
    from vader import vader_score_dataset, get_detailed_scores, analyze_sentiment_vader
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

def main():
    """
    Main function to run the sentiment analysis pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Sentiment Analysis for Movie/Product Reviews')
    parser.add_argument('--data', type=str, default='data/imdb_reviews_sample.csv', help='Path to the input data')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'vader', 'full', 'web'], default='full',
                        help='Mode to run the application in')
    parser.add_argument('--text', type=str, help='Text to classify (for predict mode)')
    parser.add_argument('--model', type=str, 
                       choices=['logistic_regression', 'svm', 'random_forest', 'gradient_boosting', 'xgboost', 'ensemble'], 
                       default='ensemble', help='Model to use for prediction')
    parser.add_argument('--cache', action='store_true', help='Use cached preprocessed data and models when available')
    parser.add_argument('--larger_dataset', action='store_true', help='Use the larger dataset in data/larger_dataset/')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel jobs for preprocessing')
    
    args = parser.parse_args()
    
    # Make sure the necessary directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Determine the dataset to use
    if args.larger_dataset:
        args.data = 'data/larger_dataset/full_dataset.csv'
        logger.info(f"Using larger dataset from {args.data}")
    
    # Load data if not in predict mode with a specific text
    df = None
    if args.mode != 'predict' or args.text is None:
        try:
            logger.info(f"Loading data from {args.data}...")
            df = load_data(args.data)
            logger.info(f"Loaded {len(df)} records")
            
            # Display sample data
            logger.info("\nSample data:")
            sample_data = display_sample_records(df)
            print(sample_data)
            
            # Plot class distribution
            plot_class_distribution(df)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(traceback.format_exc())
            if args.mode != 'predict':
                return
    
    # Training mode
    model_data = {}
    processed_df = None
    if args.mode == 'train' or args.mode == 'full':
        logger.info("\n--- Starting Training Pipeline ---")
        
        try:
            # Preprocess the data with error handling
            logger.info("Preprocessing data...")
            processed_df = preprocess_dataframe(
                df, 
                handle_neg=True, 
                use_pos_tagging=False, 
                extract_features=True, 
                n_jobs=args.parallel
            )
            
            # Display sample preprocessed data
            logger.info("\nSample preprocessed data:")
            sample_data = display_sample_records(processed_df)
            print(sample_data[['review', 'processed_text']])
            
            # Plot word frequencies
            plot_word_frequencies(processed_df['processed_text'])
            
            # Train models with improved efficiency
            logger.info("\nTraining models...")
            model_data = train_models(processed_df, use_cached=args.cache)
            
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error in training phase: {e}")
            logger.error(traceback.format_exc())
            if args.mode == 'train':
                return
    
    # Evaluation mode
    if args.mode == 'evaluate' or args.mode == 'full':
        try:
            # Check if we have model data from training, otherwise load models
            if args.mode == 'evaluate' and not model_data:
                logger.info("\n--- Starting Evaluation Pipeline ---")
                
                # Preprocess the data if not already done
                if processed_df is None and df is not None:
                    logger.info("Preprocessing data...")
                    processed_df = preprocess_dataframe(
                        df, 
                        handle_neg=True, 
                        use_pos_tagging=False, 
                        extract_features=True, 
                        n_jobs=args.parallel
                    )
                
                # Load models
                logger.info(f"Loading model: {args.model}")
                model = load_model(args.model)
                vectorizer = load_vectorizer(args.model)
                
                # Prepare test data
                X, _ = build_feature_matrix(processed_df, vectorizer, fit=False)
                y = processed_df['sentiment']
                
                model_data = {
                    args.model: model,
                    'X_test': X,
                    'y_test': y,
                    'X_test_raw': processed_df['processed_text'],
                    'vectorizer': vectorizer
                }
            
            # Skip evaluation if we don't have model data
            if not model_data:
                logger.warning("No model data available for evaluation. Skipping.")
                if args.mode == 'evaluate':
                    return
            
            # Evaluate models
            logger.info("\nEvaluating models...")
            metrics_list = []
            
            # Define class names for confusion matrix
            if 'y_test' in model_data:
                classes = sorted(set(model_data['y_test']))
            
                # Determine which models to evaluate
                if args.mode == 'full':
                    models_to_evaluate = [
                        model_name for model_name in 
                        ['logistic_regression', 'svm', 'random_forest', 'gradient_boosting', 'xgboost', 'ensemble'] 
                        if model_name in model_data
                    ]
                else:
                    models_to_evaluate = [args.model]
                
                for model_name in models_to_evaluate:
                    if model_name in model_data:
                        logger.info(f"\n{model_name.replace('_', ' ').title()} Evaluation:")
                        metrics = evaluate_model(
                            model_data[model_name], 
                            model_data['X_test'], 
                            model_data['y_test'], 
                            model_name.replace('_', ' ').title()
                        )
                        metrics_list.append(metrics)
                        
                        # Plot confusion matrix
                        plot_confusion_matrix(
                            model_data['y_test'], 
                            metrics['y_pred'], 
                            classes, 
                            model_name.title()
                        )
                        
                        # Plot ROC curve if probabilities are available
                        if metrics['y_prob'] is not None:
                            plot_roc_curve(
                                model_data['y_test'],
                                metrics['y_prob'],
                                classes,
                                model_name.title()
                            )
                        
                        # Analyze misclassifications
                        analyze_misclassifications(
                            model_data['X_test_raw'],
                            model_data['y_test'],
                            metrics['y_pred'],
                            classes,
                            model_name.title()
                        )
                        
                        # Plot feature importance for supported models
                        if model_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost']:
                            try:
                                if hasattr(model_data['vectorizer'], 'get_feature_names_out'):
                                    feature_names = model_data['vectorizer'].get_feature_names_out()
                                else:
                                    feature_names = [f'feature_{i}' for i in range(100)]  # Fallback if feature names not available
                                    
                                plot_feature_importance(
                                    model_data[model_name],
                                    feature_names[:100],  # Limit to top 100 features for visualization
                                    model_name.title()
                                )
                            except Exception as e:
                                logger.error(f"Could not generate feature importance: {e}")
                
                # Save and plot metrics comparison
                if len(metrics_list) > 0:
                    metrics_df = save_metrics_to_csv(metrics_list)
                    plot_metrics_comparison(metrics_df)
                    
                    # Create interactive dashboard
                    create_interactive_dashboard(metrics_list)
        except Exception as e:
            logger.error(f"Error in evaluation phase: {e}")
            logger.error(traceback.format_exc())
            if args.mode == 'evaluate':
                return
    
    # VADER analysis mode
    if args.mode == 'vader' or args.mode == 'full':
        try:
            logger.info("\n--- Starting VADER Sentiment Analysis ---")
            
            # Skip VADER analysis if we don't have data
            if df is None:
                logger.warning("No data available for VADER analysis. Skipping.")
                if args.mode == 'vader':
                    return
            
            # Apply VADER analysis
            vader_results = vader_score_dataset(df)
            
            # Save results
            save_data(vader_results, 'results/vader_results.csv')
            
            # Get agreement with labeled data
            if model_data and 'y_test' in model_data:
                # Compare with model predictions
                logger.info("\nComparing VADER with ML models:")
                
                # Get relevant model predictions
                model_preds = {}
                for model_name in model_data:
                    if model_name in ['logistic_regression', 'svm', 'random_forest', 'gradient_boosting', 'xgboost', 'ensemble']:
                        model_preds[model_name] = model_data[model_name].predict(model_data['X_test'])
                
                # Convert to DataFrame for easy comparison
                comparison_df = pd.DataFrame({
                    'text': model_data['X_test_raw'],
                    'true_sentiment': model_data['y_test']
                })
                
                # Add model predictions
                for model_name, preds in model_preds.items():
                    comparison_df[model_name] = preds
                
                # Add VADER sentiment
                try:
                    if hasattr(vader_results, 'loc') and hasattr(model_data['y_test'], 'index'):
                        comparison_df['vader'] = vader_results.loc[model_data['y_test'].index, 'vader_sentiment'].values
                    else:
                        # Fallback if index-based lookup doesn't work
                        vader_sentiments = []
                        for text in comparison_df['text']:
                            vader_sentiment = analyze_sentiment_vader(text)
                            vader_sentiments.append(vader_sentiment)
                        comparison_df['vader'] = vader_sentiments
                
                    # Calculate agreement rates
                    for model_name in model_preds:
                        agreement = (comparison_df['true_sentiment'] == comparison_df[model_name]).mean()
                        logger.info(f"{model_name} agreement: {agreement:.4f}")
                    
                    vader_agreement = (comparison_df['true_sentiment'] == comparison_df['vader']).mean()
                    logger.info(f"VADER agreement: {vader_agreement:.4f}")
                    
                    # Save comparison
                    save_data(comparison_df, 'results/model_comparison.csv')
                except Exception as e:
                    logger.error(f"Error comparing VADER with models: {e}")
        except Exception as e:
            logger.error(f"Error in VADER analysis phase: {e}")
            logger.error(traceback.format_exc())
    
    # Prediction mode for a specific text
    if args.mode == 'predict' and args.text is not None:
        try:
            logger.info(f"\n--- Predicting Sentiment for Text ---")
            
            # Load model and vectorizer
            model = load_model(args.model)
            vectorizer = load_vectorizer(args.model)
            
            # Preprocess the input text
            processed_text = preprocess_text(args.text, handle_neg=True)
            
            # Vectorize the text
            text_vec = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vec)[0]
            
            # Get prediction probability if available
            probability = None
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(text_vec)[0]
                max_proba_idx = probas.argmax()
                probability = probas[max_proba_idx]
            
            print(f"\nInput text: {args.text}")
            print(f"Preprocessed: {processed_text}")
            print(f"Predicted sentiment ({args.model}): {prediction}")
            if probability:
                print(f"Confidence: {probability:.2f}")
            
            # Also get VADER prediction
            vader_prediction = analyze_sentiment_vader(args.text)
            vader_scores = get_detailed_scores([args.text]).iloc[0].to_dict()
            print(f"VADER sentiment: {vader_prediction}")
            print(f"VADER scores: {vader_scores}")
            
            # Create an HTML snippet for the result
            result_html = f"""
            <div class="result-card {'positive' if prediction == 'positive' else 'negative' if prediction == 'negative' else 'neutral'}">
                <h3>New Prediction Result</h3>
                <p><strong>Input:</strong> "{args.text}"</p>
                <p><strong>Processed:</strong> {processed_text}</p>
                <p><strong>{args.model.replace('_', ' ').title()}:</strong> {prediction} {f'({probability:.2f} confidence)' if probability else ''}</p>
                <p><strong>VADER:</strong> {vader_prediction} (compound score: {vader_scores['compound']:.2f})</p>
            </div>
            """
            
            # Append this result to the HTML dashboard
            try:
                if os.path.exists('results/sentiment_results.html'):
                    with open('results/sentiment_results.html', 'r') as f:
                        html_content = f.read()
                        
                    # Find position to insert before the footer
                    insert_pos = html_content.find("<footer>")
                    if insert_pos > 0:
                        new_html = html_content[:insert_pos] + result_html + html_content[insert_pos:]
                        
                        with open('results/sentiment_results.html', 'w') as f:
                            f.write(new_html)
                        
                        print("\nResult added to the dashboard.")
            except Exception as e:
                logger.error(f"Could not update dashboard: {e}")
        except Exception as e:
            logger.error(f"Error in prediction phase: {e}")
            logger.error(traceback.format_exc())
    
    # Web UI mode (start Flask server)
    if args.mode == 'web':
        try:
            logger.info("\n--- Starting Web Interface ---")
            
            # Import the Flask app
            try:
                from app import app
                
                # Start the Flask server
                app.run(host='0.0.0.0', port=5000, debug=True)
            except ImportError:
                from src.app import app
                
                # Start the Flask server
                app.run(host='0.0.0.0', port=5000, debug=True)
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 