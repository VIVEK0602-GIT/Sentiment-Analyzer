import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter

def load_data(path):
    """
    Load data from CSV file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file {path} not found")
    
    return pd.read_csv(path)

def save_data(df, path, index=False):
    """
    Save DataFrame to CSV file
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=index)
    print(f"Data saved to {path}")

def create_timestamp():
    """
    Create a timestamp string for file naming
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_experiment_id():
    """
    Generate a unique experiment ID
    """
    timestamp = create_timestamp()
    random_suffix = ''.join(random.choices('0123456789ABCDEF', k=4))
    return f"exp_{timestamp}_{random_suffix}"

def plot_class_distribution(df, label_column='sentiment', title='Class Distribution'):
    """
    Plot the distribution of class labels
    """
    counts = df[label_column].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'results/class_distribution.png')
    plt.close()
    
    return counts

def display_sample_records(df, n=5):
    """
    Display a random sample of n records
    """
    return df.sample(n)

def get_most_common_words(texts, n=20):
    """
    Get the most common words in a corpus
    """
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    return Counter(all_words).most_common(n)

def plot_word_frequencies(texts, n=20, title='Most Common Words'):
    """
    Plot the frequencies of the most common words
    """
    word_freq = get_most_common_words(texts, n)
    
    words = [item[0] for item in word_freq]
    freqs = [item[1] for item in word_freq]
    
    plt.figure(figsize=(12, 8))
    plt.bar(words, freqs)
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'results/word_frequencies.png')
    plt.close()
    
    return word_freq