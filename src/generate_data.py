import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from nltk.corpus import movie_reviews
import nltk
from utils import save_data
import random

# Download necessary NLTK resources
nltk.download('movie_reviews')

def generate_large_dataset(output_path, num_samples=5000):
    """
    Generate a larger dataset by augmenting existing samples
    and including additional synthetic data
    """
    print("Loading existing data...")
    
    # Load existing data - fix the path to be relative to the project root
    existing_data = pd.read_csv('../data/imdb_reviews_sample.csv')
    
    # Create templates for synthetic data generation
    positive_templates = [
        "I really enjoyed {item}. {reason}",
        "{item} was fantastic! {reason}",
        "I loved {item}. {reason}",
        "Highly recommend {item}. {reason}",
        "{item} exceeded my expectations. {reason}",
        "Amazing experience with {item}. {reason}",
        "Best {category} I've ever {action}. {reason}",
        "{item} is outstanding. {reason}",
        "I'm very impressed with {item}. {reason}",
        "Five stars for {item}! {reason}"
    ]
    
    negative_templates = [
        "I was disappointed by {item}. {reason}",
        "{item} was terrible. {reason}",
        "I regret {action} {item}. {reason}",
        "Would not recommend {item}. {reason}",
        "{item} fell short of expectations. {reason}",
        "Awful experience with {item}. {reason}",
        "Worst {category} I've ever {action}. {reason}",
        "{item} is frustrating. {reason}",
        "Don't waste your time on {item}. {reason}",
        "One star for {item}. {reason}"
    ]
    
    neutral_templates = [
        "{item} was average. {reason}",
        "Not bad, not great. {item} is just okay. {reason}",
        "Mixed feelings about {item}. {reason}",
        "{item} has its pros and cons. {reason}",
        "Somewhat satisfied with {item}. {reason}",
        "{item} is mediocre at best. {reason}",
        "Middle-of-the-road {category}. {reason}",
        "{item} didn't wow me but wasn't terrible. {reason}",
        "Three stars for {item}. {reason}",
        "I'm indifferent about {item}. {reason}"
    ]
    
    # Lists of items to fill in templates
    items = [
        "the movie", "this film", "the product", "this show", "the documentary",
        "the series", "this book", "the restaurant", "this hotel", "the service",
        "the album", "this app", "the game", "this device", "the experience"
    ]
    
    categories = [
        "movie", "film", "product", "show", "documentary", "series", "book", 
        "restaurant", "hotel", "service", "album", "app", "game", "device", "experience"
    ]
    
    actions = [
        "watched", "used", "purchased", "tried", "experienced", "bought", "seen", 
        "heard", "visited", "played", "listened to", "read", "downloaded"
    ]
    
    positive_reasons = [
        "The quality was exceptional.",
        "It had amazing features.",
        "The performance was outstanding.",
        "I couldn't ask for anything better.",
        "It exceeded all my expectations.",
        "Every aspect was perfectly executed.",
        "The attention to detail was impressive.",
        "I was completely satisfied with everything.",
        "It provided incredible value for the money.",
        "The experience was flawless from start to finish."
    ]
    
    negative_reasons = [
        "The quality was poor.",
        "It lacked basic features.",
        "The performance was disappointing.",
        "There were too many issues to count.",
        "It failed to meet even my lowest expectations.",
        "Nothing worked as it should have.",
        "There was a clear lack of attention to detail.",
        "I was completely dissatisfied with everything.",
        "It was a waste of money.",
        "The experience was frustrating from start to finish."
    ]
    
    neutral_reasons = [
        "Some aspects were good, others were not.",
        "It had a mix of useful and pointless features.",
        "The performance was inconsistent.",
        "It met some expectations but missed others.",
        "Some parts were well executed, others weren't.",
        "It had both pros and cons.",
        "It was neither impressive nor disappointing.",
        "It was functional but not exceptional.",
        "It was worth the price, but just barely.",
        "The experience had both high and low points."
    ]
    
    # Generate synthetic reviews
    synthetic_reviews = []
    
    # Calculate how many reviews of each type to generate
    total_needed = num_samples - len(existing_data)
    per_sentiment = total_needed // 3
    
    print(f"Generating {total_needed} synthetic reviews...")
    
    # Generate positive reviews
    for _ in range(per_sentiment):
        template = random.choice(positive_templates)
        item = random.choice(items)
        category = random.choice(categories)
        action = random.choice(actions)
        reason = random.choice(positive_reasons)
        
        review = template.format(item=item, category=category, action=action, reason=reason)
        synthetic_reviews.append({"review": review, "sentiment": "positive"})
    
    # Generate negative reviews
    for _ in range(per_sentiment):
        template = random.choice(negative_templates)
        item = random.choice(items)
        category = random.choice(categories)
        action = random.choice(actions)
        reason = random.choice(negative_reasons)
        
        review = template.format(item=item, category=category, action=action, reason=reason)
        synthetic_reviews.append({"review": review, "sentiment": "negative"})
    
    # Generate neutral reviews
    for _ in range(per_sentiment):
        template = random.choice(neutral_templates)
        item = random.choice(items)
        category = random.choice(categories)
        action = random.choice(actions)
        reason = random.choice(neutral_reasons)
        
        review = template.format(item=item, category=category, action=action, reason=reason)
        synthetic_reviews.append({"review": review, "sentiment": "neutral"})
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_reviews)
    
    # Combine existing and synthetic data
    combined_df = pd.concat([existing_data, synthetic_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])
    
    # Ensure output directory exists - fix the path to be relative to the project root
    full_output_path = os.path.join('..', output_path)
    os.makedirs(full_output_path, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(full_output_path, 'train.csv')
    val_path = os.path.join(full_output_path, 'validation.csv')
    test_path = os.path.join(full_output_path, 'test.csv')
    full_path = os.path.join(full_output_path, 'full_dataset.csv')
    
    save_data(train_df, train_path)
    save_data(val_df, val_path)
    save_data(test_df, test_path)
    save_data(combined_df, full_path)
    
    print(f"Generated dataset with {len(combined_df)} reviews")
    print(f"Training set: {len(train_df)} reviews")
    print(f"Validation set: {len(val_df)} reviews")
    print(f"Test set: {len(test_df)} reviews")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'full': combined_df
    }

if __name__ == "__main__":
    output_path = 'data/larger_dataset'
    generate_large_dataset(output_path, num_samples=5000) 