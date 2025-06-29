import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import wordnet
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import functools
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    # Attempt to download necessary NLTK resources but handle failures gracefully
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet data
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")
    print("Falling back to simplified processing...")

# Create a cache directory
os.makedirs('cache', exist_ok=True)

# Add a caching decorator for expensive operations
def memoize(func):
    """Cache the results of the function"""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def clean_text(text, lemmatize=True):
    """
    Basic text cleaning: convert to lowercase, remove special characters and numbers
    """
    # Handle missing/NaN values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize if requested
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    # Join tokens back into a string
    return ' '.join(tokens)

def remove_stopwords(tokens, extra_stopwords=None):
    """
    Remove stopwords from tokenized text
    
    Parameters:
    -----------
    tokens : list
        List of tokens
    extra_stopwords : list, optional
        Additional stopwords to remove
        
    Returns:
    --------
    list
        Tokens with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback if NLTK resources aren't available
        stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
            't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])
    
    # Add extra stopwords if provided
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    
    return [t for t in tokens if t not in stop_words]

def get_wordnet_pos(word):
    """
    Map POS tag to first character used by WordNetLemmatizer
    """
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)
    except:
        return wordnet.NOUN

def lemmatize_text(tokens, use_pos=False):
    """
    Lemmatize tokens to their root form with optional POS tagging
    """
    try:
        lemmatizer = WordNetLemmatizer()
        
        if use_pos:
            return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
        else:
            return [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # Fallback when NLTK resources aren't available
        return tokens

def stem_text(tokens):
    """
    Stem tokens to their root form using Porter Stemmer
    """
    try:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    except:
        # Fallback when NLTK resources aren't available
        return tokens

def remove_rare_words(tokens, min_count=2, corpus_tokens=None):
    """
    Remove words that appear less than min_count times in the corpus
    
    Parameters:
    -----------
    tokens : list
        List of tokens
    min_count : int
        Minimum count threshold
    corpus_tokens : list, optional
        All tokens in the corpus, if not provided will only consider current tokens
        
    Returns:
    --------
    list
        Tokens with rare words removed
    """
    if corpus_tokens is None:
        corpus_tokens = tokens
    
    # Count word frequencies
    word_counts = Counter(corpus_tokens)
    
    # Keep only words that appear at least min_count times
    return [t for t in tokens if word_counts[t] >= min_count]

def handle_negation(tokens):
    """
    Enhanced negation handling in text by adding NOT_ prefix to words following a negation
    
    Parameters:
    -----------
    tokens : list
        List of tokens
        
    Returns:
    --------
    list
        Tokens with negation handled
    """
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing', 'nowhere', 
                      'hardly', 'scarcely', 'barely', 'doesnt', 'isnt', 'wasnt', 'shouldnt', 
                      'wouldnt', 'couldnt', 'wont', 'cant', 'dont', 'arent', 'aint'}
    
    negation_enders = {'.', '!', '?', ',', ';', ':', 'but', 'however', 'nevertheless', 'nonetheless', 'although', 'though', 'despite', 'whereas'}
    
    intensifiers = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally'}
    
    result = []
    negated = False
    negation_distance = 0
    max_negation_distance = 5  # Maximum words to apply negation
    
    for token in tokens:
        if token in negation_words:
            negated = True
            negation_distance = 0
            result.append(token)
        elif token in negation_enders:
            negated = False
            result.append(token)
        elif negated and negation_distance < max_negation_distance:
            # Handle intensifiers by preserving them but continuing negation scope
            if token in intensifiers:
                result.append(token)
            else:
                result.append('NOT_' + token)
                negation_distance += 1
        else:
            result.append(token)
            if negated:
                negation_distance += 1
    
    return result

def extract_text_features(text):
    """
    Extract features from text such as text length, average word length, etc.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    if pd.isna(text) or text is None or not isinstance(text, str):
        text = ""
    
    # Tokenize text
    tokens = text.split()
    
    # Extract features
    features = {
        'text_length': len(text),
        'word_count': len(tokens),
        'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
        'special_char_count': sum(1 for char in text if char in string.punctuation),
        'uppercase_count': sum(1 for char in text if char.isupper()),
        'digit_count': sum(1 for char in text if char.isdigit()),
        'unique_word_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'sentiment_words': count_sentiment_words(text)
    }
    
    return features

# Add sentiment word counting function
def count_sentiment_words(text):
    """Count common sentiment words in text"""
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'awesome', 
                     'fantastic', 'perfect', 'happy', 'enjoyed', 'positive', 'recommend', 'impressed'}
    
    negative_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor', 'hate', 'disappointing', 
                     'waste', 'negative', 'boring', 'disappointed', 'failure', 'fail', 'mediocre'}
    
    # Create pattern for efficient search
    pos_pattern = r'\b(' + '|'.join(positive_words) + r')\b'
    neg_pattern = r'\b(' + '|'.join(negative_words) + r')\b'
    
    # Count occurrences
    pos_count = len(re.findall(pos_pattern, text.lower()))
    neg_count = len(re.findall(neg_pattern, text.lower()))
    
    return pos_count - neg_count

# Optimize the preprocess_text function with memoization
@memoize
def preprocess_text(text, remove_stops=True, lemmatize=True, stem=False, handle_neg=False, extra_stopwords=None, use_pos_tagging=False):
    """
    Preprocess text by applying cleaning, tokenization, and normalization steps.
    Simplified version with better efficiency.
    
    Parameters:
    -----------
    text : str
        Input text
    remove_stops : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize text
    stem : bool
        Whether to stem text
    handle_neg : bool
        Whether to handle negation
    extra_stopwords : list
        Additional stopwords to remove
    use_pos_tagging : bool
        Whether to use POS tagging for lemmatization
        
    Returns:
    --------
    str
        Preprocessed text
    """
    # Handle missing/NaN values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Fast tokenization with regex
    tokens = re.findall(r'\b\w+\b', cleaned_text)
    
    # Handle negation if requested
    if handle_neg:
        tokens = handle_negation(tokens)
    
    # Remove stopwords if requested
    if remove_stops:
        tokens = remove_stopwords(tokens, extra_stopwords)
    
    # Lemmatize if requested (but avoid POS tagging for efficiency)
    if lemmatize:
        tokens = lemmatize_text(tokens, use_pos=False)
    
    # Stem if requested (note: typically you would use either lemmatization or stemming, not both)
    if stem:
        tokens = stem_text(tokens)
    
    # Join tokens back into a string
    return ' '.join(tokens)

# Optimize dataframe preprocessing with parallel processing
def preprocess_dataframe(df, text_column='review', remove_stops=True, lemmatize=True, stem=False, 
                         handle_neg=False, extra_stopwords=None, use_pos_tagging=False, 
                         extract_features=False, n_jobs=4, batch_size=100):
    """
    Preprocess a dataframe containing text data
    This optimized version uses parallel processing for faster execution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Column containing text to preprocess
    remove_stops : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize text
    stem : bool
        Whether to stem text
    handle_neg : bool
        Whether to handle negation
    extra_stopwords : list
        Additional stopwords to remove
    use_pos_tagging : bool
        Whether to use POS tagging for lemmatization
    extract_features : bool
        Whether to extract additional text features
    n_jobs : int
        Number of parallel jobs to run (smaller value may avoid NLTK resource issues)
    batch_size : int
        Batch size for parallel processing
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with preprocessed text
    """
    # Check if cached result exists
    cache_file = f'cache/preprocessed_df_{text_column}_{remove_stops}_{lemmatize}_{stem}_{handle_neg}_{use_pos_tagging}.pkl'
    
    if os.path.exists(cache_file):
        print(f"Loading preprocessed data from cache: {cache_file}")
        return joblib.load(cache_file)
    
    print("Preprocessing data with parallel processing...")
    
    # Make a copy of the dataframe
    result_df = df.copy()
    
    # Define preprocessing function for a single text
    def preprocess_single(text):
        return preprocess_text(
            text, 
            remove_stops=remove_stops,
            lemmatize=lemmatize,
            stem=stem,
            handle_neg=handle_neg,
            extra_stopwords=extra_stopwords,
            use_pos_tagging=False  # Always disable POS tagging to avoid NLTK issues
        )
    
    # For smaller datasets, serial processing can be more efficient to avoid overhead
    if len(df) < 100:
        print("Small dataset detected, using serial processing...")
        processed_texts = [preprocess_single(text) for text in df[text_column].fillna("")]
    else:
        # Process texts in parallel with progress bar
        try:
            texts = df[text_column].fillna("").tolist()
            processed_texts = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
                delayed(preprocess_single)(text) for text in tqdm(texts, desc="Preprocessing texts")
            )
        except Exception as e:
            print(f"Warning: Parallel processing failed ({e}), falling back to serial processing...")
            processed_texts = [preprocess_single(text) for text in tqdm(df[text_column].fillna(""), desc="Preprocessing texts")]
    
    # Add processed text to dataframe
    result_df['processed_text'] = processed_texts
    
    # Extract additional features if requested
    if extract_features:
        print("Extracting text features...")
        
        try:
            feature_dicts = []
            for text in tqdm(df[text_column].fillna(""), desc="Extracting features"):
                feature_dicts.append(extract_text_features(text))
            
            # Convert list of dictionaries to dataframe
            features_df = pd.DataFrame(feature_dicts)
            
            # Combine with original dataframe
            for col in features_df.columns:
                result_df[f'feature_{col}'] = features_df[col]
        except Exception as e:
            print(f"Warning: Feature extraction failed ({e}), skipping feature extraction...")
    
    # Cache the result
    try:
        joblib.dump(result_df, cache_file)
        print(f"Preprocessed data saved to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to cache preprocessed data ({e})")
    
    return result_df

def create_vectorizers(max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.95):
    """
    Create improved TF-IDF and Count vectorizers
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b\w+\b'
    )
    
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b\w+\b'
    )
    
    return tfidf_vectorizer, count_vectorizer

def extract_features(texts, vectorizer, fit=True):
    """
    Extract features from preprocessed texts using the specified vectorizer
    
    Parameters:
    -----------
    texts : list or pandas.Series
        Input texts
    vectorizer : sklearn vectorizer
        Vectorizer to use
    fit : bool
        Whether to fit the vectorizer on the texts or just transform
        
    Returns:
    --------
    scipy.sparse.csr.csr_matrix
        Extracted features
    """
    if fit:
        return vectorizer.fit_transform(texts)
    else:
        return vectorizer.transform(texts)

def generate_wordcloud(texts, sentiment=None, save_path=None, figsize=(12, 8), max_words=100):
    """
    Generate a word cloud from texts
    
    Parameters:
    -----------
    texts : list or pandas.Series
        Input texts
    sentiment : str, optional
        Sentiment label for the title
    save_path : str, optional
        Path to save the word cloud image
    figsize : tuple
        Figure size
    max_words : int
        Maximum number of words in the word cloud
        
    Returns:
    --------
    matplotlib.figure.Figure
        Word cloud figure
    """
    # Combine all texts
    if isinstance(texts, pd.Series):
        all_text = ' '.join(texts)
    else:
        all_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=max_words,
        contour_width=3
    ).generate(all_text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if sentiment:
        plt.title(f'Word Cloud - {sentiment.capitalize()} Reviews', fontsize=18)
    
    # Save if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

def plot_word_frequencies_by_sentiment(df, text_column='processed_text', sentiment_column='sentiment', 
                                       top_n=20, save_path=None, figsize=(15, 12)):
    """
    Plot word frequencies separated by sentiment
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Name of the column containing preprocessed text
    sentiment_column : str
        Name of the column containing sentiment labels
    top_n : int
        Number of top words to show
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Word frequencies figure
    """
    # Get unique sentiment values
    sentiments = df[sentiment_column].unique()
    
    # Create subplots
    fig, axes = plt.subplots(len(sentiments), 1, figsize=figsize)
    
    if len(sentiments) == 1:
        axes = [axes]
    
    # Plot for each sentiment
    for i, sentiment in enumerate(sentiments):
        # Filter dataframe for the current sentiment
        sentiment_df = df[df[sentiment_column] == sentiment]
        
        # Get all words
        all_words = []
        for text in sentiment_df[text_column]:
            all_words.extend(text.split())
        
        # Count word frequencies
        word_counts = Counter(all_words).most_common(top_n)
        
        # Extract words and frequencies
        words = [item[0] for item in word_counts]
        frequencies = [item[1] for item in word_counts]
        
        # Plot
        axes[i].barh(range(len(words)), frequencies, align='center')
        axes[i].set_yticks(range(len(words)))
        axes[i].set_yticklabels(words)
        axes[i].invert_yaxis()
        axes[i].set_xlabel('Frequency')
        axes[i].set_title(f'Top {top_n} Words in {sentiment.capitalize()} Reviews')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return fig

def get_preprocessor(lemmatize=True):
    return lambda text: clean_text(text, lemmatize=lemmatize)