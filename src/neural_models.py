import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json

class NeuralSentimentClassifier:
    def __init__(self, model_type='lstm', embedding_dim=100, max_features=10000, max_len=200):
        """
        Initialize a neural network model for sentiment analysis
        
        Parameters:
        -----------
        model_type : str
            Type of neural network to use ('cnn', 'lstm', 'bilstm')
        embedding_dim : int
            Dimension of word embeddings
        max_features : int
            Maximum number of words to include in the vocabulary
        max_len : int
            Maximum length of sequences
        """
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.history = None
    
    def preprocess_data(self, texts, labels=None, train=False):
        """
        Preprocess the text data for neural networks
        
        Parameters:
        -----------
        texts : list or pandas Series
            The text data to preprocess
        labels : list or pandas Series, optional
            The labels for the text data
        train : bool
            Whether this is training data (to fit tokenizer and label encoder)
            
        Returns:
        --------
        X : numpy array
            Preprocessed text data
        y : numpy array, optional
            Encoded labels (if labels are provided)
        """
        # Convert texts to list if it's a pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenize the texts
        if train or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features)
            self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to the same length
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Process labels if provided
        if labels is not None:
            if isinstance(labels, pd.Series):
                labels = labels.tolist()
            
            if train or self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(labels)
            
            y = self.label_encoder.transform(labels)
            
            # Convert to categorical if more than 2 classes
            if len(self.label_encoder.classes_) > 2:
                y = tf.keras.utils.to_categorical(y)
            
            return X, y
        
        return X
    
    def build_model(self, num_classes):
        """
        Build the neural network model
        
        Parameters:
        -----------
        num_classes : int
            Number of classes for classification
            
        Returns:
        --------
        model : keras.Model
            The built model
        """
        # Determine the output layer and loss function based on number of classes
        if num_classes == 2:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_units = num_classes
            activation = 'softmax'
            loss = 'categorical_crossentropy'
        
        # Build model based on the specified type
        if self.model_type == 'cnn':
            model = Sequential([
                Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
                Conv1D(128, 5, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(output_units, activation=activation)
            ])
        
        elif self.model_type == 'lstm':
            model = Sequential([
                Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(output_units, activation=activation)
            ])
        
        elif self.model_type == 'bilstm':
            model = Sequential([
                Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
                Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(output_units, activation=activation)
            ])
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, model_path='../models'):
        """
        Train the neural network model
        
        Parameters:
        -----------
        X_train : numpy array
            Training data
        y_train : numpy array
            Training labels
        X_val : numpy array, optional
            Validation data
        y_val : numpy array, optional
            Validation labels
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size for training
        model_path : str
            Path to save the model
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # If model doesn't exist, build it
        if self.model is None:
            num_classes = len(self.label_encoder.classes_)
            self.build_model(num_classes)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(model_path, f'{self.model_type}_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        self.history = history.history
        
        # Save the tokenizer and label encoder
        joblib.dump(self.tokenizer, os.path.join(model_path, f'{self.model_type}_tokenizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_path, f'{self.model_type}_label_encoder.pkl'))
        
        # Save the model configuration
        config = {
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'max_features': self.max_features,
            'max_len': self.max_len,
            'num_classes': len(self.label_encoder.classes_),
            'class_labels': self.label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(model_path, f'{self.model_type}_config.json'), 'w') as f:
            json.dump(config, f)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X_test : numpy array
            Test data
        y_test : numpy array
            Test labels
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def predict(self, texts):
        """
        Predict sentiment for new texts
        
        Parameters:
        -----------
        texts : list or pandas Series
            The text data to predict sentiment for
            
        Returns:
        --------
        predictions : numpy array
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess the texts
        X = self.preprocess_data(texts)
        
        # Get predictions
        preds = self.model.predict(X)
        
        # Convert predictions to class labels
        if len(self.label_encoder.classes_) > 2:
            pred_classes = np.argmax(preds, axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()
        
        return self.label_encoder.inverse_transform(pred_classes)
    
    @classmethod
    def load_model(cls, model_type='lstm', model_path='../models'):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_type : str
            Type of neural network to load ('cnn', 'lstm', 'bilstm')
        model_path : str
            Path where the model is saved
            
        Returns:
        --------
        classifier : NeuralSentimentClassifier
            The loaded classifier
        """
        # Check if model files exist
        model_file = os.path.join(model_path, f'{model_type}_model.h5')
        tokenizer_file = os.path.join(model_path, f'{model_type}_tokenizer.pkl')
        label_encoder_file = os.path.join(model_path, f'{model_type}_label_encoder.pkl')
        config_file = os.path.join(model_path, f'{model_type}_config.json')
        
        if not os.path.exists(model_file) or not os.path.exists(tokenizer_file) or not os.path.exists(label_encoder_file):
            raise FileNotFoundError(f"Model files for {model_type} not found in {model_path}")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create a new classifier with the same configuration
        classifier = cls(
            model_type=config['model_type'],
            embedding_dim=config['embedding_dim'],
            max_features=config['max_features'],
            max_len=config['max_len']
        )
        
        # Load tokenizer and label encoder
        classifier.tokenizer = joblib.load(tokenizer_file)
        classifier.label_encoder = joblib.load(label_encoder_file)
        
        # Build and load the model
        classifier.build_model(config['num_classes'])
        classifier.model.load_weights(model_file)
        
        return classifier
    
    def plot_training_history(self, figsize=(12, 5), save_path=None):
        """
        Plot the training history
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Train')
        if 'val_accuracy' in self.history:
            ax1.plot(self.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Train')
        if 'val_loss' in self.history:
            ax2.plot(self.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 