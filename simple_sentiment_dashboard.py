import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import webbrowser

# Get the src directory in the path
sys.path.append(os.path.abspath("."))

try:
    from src.vader import analyze_sentiment_vader, get_detailed_scores, vader_score_dataset
    from src.utils import load_data, save_data
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def analyze_single_text(text):
    """Analyze a single text input using VADER"""
    sentiment = analyze_sentiment_vader(text)
    scores = get_detailed_scores([text]).iloc[0].to_dict()
    
    print("\n=== Sentiment Analysis Results ===")
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Compound Score: {scores['compound']:.4f}")
    print(f"Positive: {scores['pos']:.4f}")
    print(f"Neutral: {scores['neu']:.4f}")
    print(f"Negative: {scores['neg']:.4f}")
    
    return {"text": text, "sentiment": sentiment, "scores": scores}

def analyze_batch_from_file(file_path):
    """Analyze a batch of texts from a file"""
    try:
        df = load_data(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head(5))
        
        # Run VADER analysis
        print("\nRunning VADER sentiment analysis...")
        results_df = vader_score_dataset(df)
        
        # Display results summary
        sentiment_counts = results_df['vader_sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # Agreement rate if ground truth exists
        if 'sentiment' in results_df.columns:
            agreement = (results_df['sentiment'] == results_df['vader_sentiment']).mean()
            print(f"\nAgreement with labeled data: {agreement:.4f}")
        
        return results_df
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def create_results_html(results, results_df=None):
    """Create an HTML dashboard from the analysis results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Results</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; max-width: 1200px; margin: 0 auto; background-color: #f9f9f9; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ text-align: center; margin-bottom: 30px; background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .card {{ background-color: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.08); }}
            .positive {{ border-left: 5px solid #28a745; }}
            .negative {{ border-left: 5px solid #dc3545; }}
            .neutral {{ border-left: 5px solid #6c757d; }}
            .meter {{ height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; margin-bottom: 10px; }}
            .meter-bar {{ height: 100%; border-radius: 10px; }}
            .positive-bar {{ background-color: #28a745; }}
            .neutral-bar {{ background-color: #6c757d; }}
            .negative-bar {{ background-color: #dc3545; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .section {{ flex: 1; min-width: 300px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .chart-container {{ width: 100%; max-width: 800px; margin: 0 auto; }}
            footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Sentiment Analysis Results</h1>
            <p>Analysis completed on {timestamp}</p>
        </div>
    """
    
    # Add individual result if available
    if results:
        sentiment_class = results["sentiment"]
        scores = results["scores"]
        
        # Calculate percentage bars
        pos_percent = scores["pos"] * 100
        neu_percent = scores["neu"] * 100
        neg_percent = scores["neg"] * 100
        
        html_content += f"""
        <div class="card {sentiment_class}">
            <h2>Text Analysis</h2>
            <p><strong>Input Text:</strong> {results["text"]}</p>
            <p><strong>Sentiment:</strong> {sentiment_class.capitalize()}</p>
            <p><strong>Compound Score:</strong> {scores["compound"]:.4f}</p>
            
            <h3>Sentiment Breakdown</h3>
            <p><strong>Positive:</strong> {scores["pos"]:.4f}</p>
            <div class="meter">
                <div class="meter-bar positive-bar" style="width: {pos_percent}%"></div>
            </div>
            
            <p><strong>Neutral:</strong> {scores["neu"]:.4f}</p>
            <div class="meter">
                <div class="meter-bar neutral-bar" style="width: {neu_percent}%"></div>
            </div>
            
            <p><strong>Negative:</strong> {scores["neg"]:.4f}</p>
            <div class="meter">
                <div class="meter-bar negative-bar" style="width: {neg_percent}%"></div>
            </div>
        </div>
        """
    
    # Add dataset analysis if available
    if results_df is not None:
        sentiment_counts = results_df['vader_sentiment'].value_counts()
        
        # Calculate percentages for the chart
        total = len(results_df)
        positive_pct = sentiment_counts.get('positive', 0) / total * 100
        neutral_pct = sentiment_counts.get('neutral', 0) / total * 100
        negative_pct = sentiment_counts.get('negative', 0) / total * 100
        
        html_content += f"""
        <div class="card">
            <h2>Dataset Analysis</h2>
            <div class="container">
                <div class="section">
                    <h3>Sentiment Distribution</h3>
                    <div class="chart-container">
                        <div class="meter">
                            <div class="meter-bar positive-bar" style="width: {positive_pct}%"></div>
                        </div>
                        <p>Positive: {sentiment_counts.get('positive', 0)} ({positive_pct:.1f}%)</p>
                        
                        <div class="meter">
                            <div class="meter-bar neutral-bar" style="width: {neutral_pct}%"></div>
                        </div>
                        <p>Neutral: {sentiment_counts.get('neutral', 0)} ({neutral_pct:.1f}%)</p>
                        
                        <div class="meter">
                            <div class="meter-bar negative-bar" style="width: {negative_pct}%"></div>
                        </div>
                        <p>Negative: {sentiment_counts.get('negative', 0)} ({negative_pct:.1f}%)</p>
                    </div>
                </div>
                <div class="section">
                    <h3>Sample Results</h3>
                    <table>
                        <tr>
                            <th>Text</th>
                            <th>Sentiment</th>
                        </tr>
        """
        
        # Add sample rows (up to 5)
        for i, row in results_df.iterrows():
            if i >= 5:  # Limit to 5 samples
                break
            text = row['review'] if 'review' in row else ''
            text = text[:100] + "..." if len(text) > 100 else text
            sentiment = row['vader_sentiment']
            html_content += f"""
                        <tr>
                            <td>{text}</td>
                            <td>{sentiment}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </div>
        """
        
        # Add agreement details if available
        if 'sentiment' in results_df.columns and 'vader_sentiment' in results_df.columns:
            agreement = (results_df['sentiment'] == results_df['vader_sentiment']).mean()
            
            # Create agreement breakdown
            agreement_by_class = {}
            for true_sentiment in results_df['sentiment'].unique():
                class_df = results_df[results_df['sentiment'] == true_sentiment]
                class_agreement = (class_df['sentiment'] == class_df['vader_sentiment']).mean()
                agreement_by_class[true_sentiment] = class_agreement
            
            html_content += f"""
            <div class="card">
                <h2>Model Evaluation</h2>
                <p><strong>Overall Agreement:</strong> {agreement:.4f}</p>
                
                <h3>Agreement by Class</h3>
                <table>
                    <tr>
                        <th>True Sentiment</th>
                        <th>Agreement</th>
                    </tr>
            """
            
            for sentiment, agr in agreement_by_class.items():
                html_content += f"""
                    <tr>
                        <td>{sentiment}</td>
                        <td>{agr:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
    
    # Close HTML
    html_content += """
        <footer>
            <p>Generated by VADER Sentiment Analysis Tool</p>
        </footer>
    </body>
    </html>
    """
    
    # Save HTML to file
    os.makedirs('results', exist_ok=True)
    file_path = 'results/vader_dashboard.html'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nResults dashboard saved to: {file_path}")
    return file_path

def main():
    """Main function to run the script"""
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments, prompt user for input
        print("\nVADER Sentiment Analysis Dashboard")
        print("1. Analyze a single text")
        print("2. Analyze a dataset from a file")
        
        choice = input("\nSelect an option (1-2): ")
        
        if choice == '1':
            text = input("\nEnter text to analyze: ")
            results = analyze_single_text(text)
            dashboard_path = create_results_html(results)
            webbrowser.open('file://' + os.path.abspath(dashboard_path))
        
        elif choice == '2':
            default_file = 'data/imdb_reviews_sample.csv'
            file_path = input(f"\nEnter file path [{default_file}]: ") or default_file
            
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found")
                return
            
            results_df = analyze_batch_from_file(file_path)
            if results_df is not None:
                dashboard_path = create_results_html(None, results_df)
                webbrowser.open('file://' + os.path.abspath(dashboard_path))
        
        else:
            print("Invalid option")
    
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        # Analyze a file
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return
        
        results_df = analyze_batch_from_file(file_path)
        if results_df is not None:
            dashboard_path = create_results_html(None, results_df)
            webbrowser.open('file://' + os.path.abspath(dashboard_path))
    
    else:
        # Analyze a text from command line
        text = ' '.join(sys.argv[1:])
        results = analyze_single_text(text)
        dashboard_path = create_results_html(results)
        webbrowser.open('file://' + os.path.abspath(dashboard_path))

if __name__ == "__main__":
    main() 