from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score, precision_score, recall_score
import joblib
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import warnings

# Silence deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_pipeline(vectorizer, classifier):
    """
    Create a scikit-learn pipeline with vectorizer and classifier
    """
    return Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

def train_logistic_regression(X_train, y_train, max_iter=1000, C=1.0):
    """
    Train a Logistic Regression model
    """
    model = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, C=1.0):
    """
    Train a Support Vector Machine model
    """
    # Use LinearSVC for faster training on large datasets
    try:
        model = LinearSVC(C=C, class_weight='balanced', max_iter=10000, dual=False)
        model.fit(X_train, y_train)
        
        # Calibrate to get probabilities
        calibrated_model = CalibratedClassifierCV(model, cv=5)
        calibrated_model.fit(X_train, y_train)
        return calibrated_model
    except Exception as e:
        print(f"Warning: LinearSVC failed ({e}), falling back to SVC...")
        model = SVC(C=C, class_weight='balanced', probability=True, max_iter=10000)
        model.fit(X_train, y_train)
        return model

def find_best_hyperparameters(pipeline, X_train, y_train, param_grid, cv=5, scoring='f1_weighted'):
    """
    Find best hyperparameters for a pipeline using GridSearchCV
    """
    # Define multiple scoring metrics
    scoring_metrics = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    # Create stratified k-fold for better cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=skf, 
        scoring=scoring_metrics,
        refit='f1',  # Choose which metric to use for selecting the best model
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Run grid search with progress bar
    print(f"Running grid search with {len(param_grid)} parameter combinations...")
    with tqdm(total=len(param_grid) * cv) as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(1)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_}")
    
    # Plot CV results if there are multiple parameter values
    plot_grid_search_results(grid_search)
    
    return grid_search.best_estimator_

def plot_grid_search_results(grid_search):
    """
    Plot grid search results for visualization
    """
    # Check if we have results to plot
    if not hasattr(grid_search, 'cv_results_'):
        return
    
    # Extract results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Only plot if we have multiple parameter values
    param_cols = [col for col in results.columns if col.startswith('param_')]
    if len(param_cols) == 0 or results[param_cols].nunique().sum() <= len(param_cols):
        return
    
    # Create plot directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot for each parameter
    for param in param_cols:
        param_name = param.replace('param_', '')
        
        # Check if parameter has multiple values
        if results[param].nunique() <= 1:
            continue
        
        # Convert parameter values to strings for plotting
        results[param] = results[param].astype(str)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot train and test scores
        plt.plot(results[param], results['mean_test_f1'], 'o-', label='Test F1')
        plt.plot(results[param], results['mean_train_f1'], 'o-', label='Train F1')
        
        # Add labels and title
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Grid Search Results for {param_name}')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'results/plots/grid_search_{param_name}.png')
        plt.close()

def save_model(model, model_name, vectorizer=None):
    """
    Save trained model and vectorizer to disk
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, f'models/{model_name}.pkl')
    
    # Save the vectorizer if provided
    if vectorizer is not None:
        joblib.dump(vectorizer, f'models/{model_name}_vectorizer.pkl')
    
    print(f"{model_name} model saved successfully")

def load_model(model_name):
    """
    Load a trained model from disk
    """
    model_path = f'models/{model_name}.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    return joblib.load(model_path)

def load_vectorizer(model_name):
    """
    Load a trained vectorizer from disk
    """
    vectorizer_path = f'models/{model_name}_vectorizer.pkl'
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file {vectorizer_path} not found")
    
    return joblib.load(vectorizer_path)

def build_feature_matrix(df, vectorizer=None, fit=True):
    """
    Build feature matrix from text data with advanced features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    vectorizer : TfidfVectorizer, optional
        Vectorizer to use, if None a new one will be created
    fit : bool
        Whether to fit the vectorizer or use a pre-fitted one
        
    Returns:
    --------
    tuple
        (X, vectorizer) - feature matrix and vectorizer
    """
    if vectorizer is None:
        # Create an improved vectorizer with better parameters
        vectorizer = TfidfVectorizer(
            max_features=15000,  # Increased features
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,            # Reduced threshold to capture more features
            max_df=0.95,         # Increased to include more common words
            sublinear_tf=True,   # Apply sublinear TF scaling
            use_idf=True,
            strip_accents='unicode',
            analyzer='word',     # Word analyzer
            token_pattern=r'\b\w+\b'  # Simple token pattern
        )
    
    # Extract text features
    if fit:
        X_text = vectorizer.fit_transform(df['processed_text'])
    else:
        X_text = vectorizer.transform(df['processed_text'])
    
    # Add additional features if available in the DataFrame
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if feature_cols:
        # Create a dense feature matrix from the additional features
        X_features = df[feature_cols].values
        
        # If X_text is sparse, convert to dense for concatenation
        if hasattr(X_text, 'toarray'):
            # If the feature matrix is too large, apply dimensionality reduction
            if X_text.shape[1] > 1000:
                try:
                    # Use TruncatedSVD to reduce dimensions
                    print("Applying dimensionality reduction...")
                    svd = TruncatedSVD(n_components=min(500, X_text.shape[1]-1))
                    X_text_reduced = svd.fit_transform(X_text)
                    X_combined = np.hstack((X_text_reduced, X_features))
                    print(f"Reduced dimensions from {X_text.shape[1]} to {X_text_reduced.shape[1]}")
                except Exception as e:
                    print(f"Warning: Dimensionality reduction failed ({e}), using original features...")
                    X_combined = np.hstack((X_text.toarray(), X_features))
            else:
                X_combined = np.hstack((X_text.toarray(), X_features))
        else:
            X_combined = np.hstack((X_text, X_features))
            
        return X_combined, vectorizer
    
    return X_text, vectorizer

def visualize_feature_importance(model, feature_names, top_n=20, save_path='results/plots/feature_importance.png'):
    """
    Visualize feature importance for models that support it
    """
    # Check if model supports feature importance
    if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance visualization")
        return
    
    # Extract feature importance
    if hasattr(model, 'coef_'):
        # For linear models
        if len(model.coef_.shape) > 1:
            # For multiclass
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    else:
        print("Model does not have a supported feature importance attribute")
        return
    
    # Create DataFrame for visualization
    if len(importance) == len(feature_names):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Top Feature Importance')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
    else:
        print(f"Feature importance length mismatch: {len(importance)} vs {len(feature_names)}")

def train_models(df, test_size=0.2, random_state=42, use_cached=True):
    """
    Train multiple models with hyper-parameter tuning and improved vectorization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with processed text
    test_size : float
        Test size for train/test split
    random_state : int
        Random state for reproducibility
    use_cached : bool
        Whether to use cached models if available
        
    Returns:
    --------
    dict
        Dictionary containing trained models and evaluation data
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create plots directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Cache paths
    lr_model_path = 'models/logistic_regression.pkl'
    svm_model_path = 'models/svm.pkl'
    rf_model_path = 'models/random_forest.pkl'
    gb_model_path = 'models/gradient_boosting.pkl'
    xgb_model_path = 'models/xgboost.pkl'
    ensemble_model_path = 'models/ensemble.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    # Check if models exist and use_cached is True
    models_exist = (
        os.path.exists(lr_model_path) and 
        os.path.exists(svm_model_path) and
        os.path.exists(rf_model_path) and
        os.path.exists(ensemble_model_path) and
        os.path.exists(vectorizer_path)
    )
    
    if use_cached and models_exist:
        print("Loading models from cache...")
        
        try:
            with open(lr_model_path, 'rb') as f:
                lr_model = pickle.load(f)
            
            with open(svm_model_path, 'rb') as f:
                svm_model = pickle.load(f)
                
            with open(rf_model_path, 'rb') as f:
                rf_model = pickle.load(f)
            
            with open(gb_model_path, 'rb') as f:
                gb_model = pickle.load(f)
            
            try:
                with open(xgb_model_path, 'rb') as f:
                    xgb_model = pickle.load(f)
            except:
                xgb_model = None
                
            with open(ensemble_model_path, 'rb') as f:
                ensemble_model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            X, _ = build_feature_matrix(df, vectorizer, fit=False)
            y = df['sentiment']
            
            models = {
                'logistic_regression': lr_model,
                'svm': svm_model,
                'random_forest': rf_model,
                'gradient_boosting': gb_model,
                'vectorizer': vectorizer,
                'X_test': X,
                'y_test': y,
                'X_test_raw': df['processed_text']
            }
            
            if xgb_model:
                models['xgboost'] = xgb_model
                
            models['ensemble'] = ensemble_model
            
            return models
        except Exception as e:
            print(f"Error loading cached models: {e}")
            print("Training new models...")
    
    print("Training models with hyperparameter tuning...")
    
    # Split the data into training and testing sets
    X, vectorizer = build_feature_matrix(df)
    y = df['sentiment']
    
    # Save important features for later use
    feature_names = None
    if hasattr(vectorizer, 'get_feature_names_out'):
        try:
            feature_names = vectorizer.get_feature_names_out()
        except:
            pass
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Store raw text for later analysis
    if isinstance(X_train, np.ndarray) and hasattr(df, 'iloc'):
        train_indices = df.index[:len(X_train)]
        test_indices = df.index[len(X_train):]
        X_train_raw = df.iloc[train_indices]['processed_text'].values
        X_test_raw = df.iloc[test_indices]['processed_text'].values
    else:
        X_train_raw = df['processed_text'].values[:len(X_train)]
        X_test_raw = df['processed_text'].values[len(X_train):]
    
    # Define parameter grids for hyperparameter tuning
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced'],
        'max_iter': [1000]
    }
    
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear'],  # Limit to linear for faster training
        'class_weight': ['balanced']
    }
    
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    # Train Logistic Regression with grid search
    print("Training Logistic Regression with hyperparameter tuning...")
    lr_model = LogisticRegression(random_state=random_state)
    lr_grid = GridSearchCV(
        lr_model,
        lr_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=4  # Limit parallel jobs to avoid memory issues
    )
    lr_grid.fit(X_train, y_train)
    lr_model = lr_grid.best_estimator_
    
    # Visualize feature importance for logistic regression
    if feature_names is not None:
        visualize_feature_importance(
            lr_model, 
            feature_names, 
            save_path='results/plots/lr_feature_importance.png'
        )
    
    # Train SVM with grid search
    print("Training SVM with hyperparameter tuning...")
    svm_model = SVC(random_state=random_state, probability=True)
    svm_grid = GridSearchCV(
        svm_model,
        svm_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=4  # Limit parallel jobs to avoid memory issues
    )
    svm_grid.fit(X_train, y_train)
    svm_model = svm_grid.best_estimator_
    
    # Train Random Forest with grid search
    print("Training Random Forest with hyperparameter tuning...")
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_grid = GridSearchCV(
        rf_model,
        rf_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=4  # Limit parallel jobs to avoid memory issues
    )
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    
    # Visualize feature importance for random forest
    if feature_names is not None:
        visualize_feature_importance(
            rf_model, 
            feature_names, 
            save_path='results/plots/rf_feature_importance.png'
        )
    
    # Train Gradient Boosting with grid search
    print("Training Gradient Boosting with hyperparameter tuning...")
    gb_model = GradientBoostingClassifier(random_state=random_state)
    gb_grid = GridSearchCV(
        gb_model,
        gb_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=4  # Limit parallel jobs to avoid memory issues
    )
    gb_grid.fit(X_train, y_train)
    gb_model = gb_grid.best_estimator_
    
    # Visualize feature importance for gradient boosting
    if feature_names is not None:
        visualize_feature_importance(
            gb_model, 
            feature_names, 
            save_path='results/plots/gb_feature_importance.png'
        )
    
    # Try to train XGBoost model if available
    xgb_model = None
    try:
        from xgboost import XGBClassifier
        
        print("Training XGBoost model...")
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = XGBClassifier(
            random_state=random_state, 
            use_label_encoder=False, 
            eval_metric='logloss',
            objective='binary:logistic' if len(np.unique(y)) == 2 else 'multi:softprob'
        )
        
        xgb_grid = GridSearchCV(
            xgb_model,
            xgb_param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=4  # Limit parallel jobs to avoid memory issues
        )
        xgb_grid.fit(X_train, y_train)
        xgb_model = xgb_grid.best_estimator_
        
        print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
        
        # Visualize feature importance for XGBoost
        if feature_names is not None:
            visualize_feature_importance(
                xgb_model, 
                feature_names, 
                save_path='results/plots/xgb_feature_importance.png'
            )
        
        with open(xgb_model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
            
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        print("Skipping XGBoost training.")
    
    # Create a more powerful ensemble model with soft voting
    print("Training Ensemble model...")
    
    # Define base estimators for the ensemble
    estimators = [
        ('lr', lr_model),
        ('svm', svm_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ]
    
    if xgb_model:
        estimators.append(('xgb', xgb_model))
    
    # Calculate optimal weights based on individual model performance
    weights = []
    for name, model in estimators:
        try:
            cv_scores = cross_validate(model, X_train, y_train, cv=5, scoring='f1_weighted')
            mean_score = np.mean(cv_scores['test_score'])
            weights.append(max(1, int(mean_score * 10)))  # Scale weights based on performance
        except:
            weights.append(1)  # Default weight if cross-validation fails
    
    print(f"Ensemble weights: {weights}")
    
    # Create and train the voting classifier
    ensemble_model = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    ensemble_model.fit(X_train, y_train)
    
    # Print best hyperparameters
    print(f"Best Logistic Regression parameters: {lr_grid.best_params_}")
    print(f"Best SVM parameters: {svm_grid.best_params_}")
    print(f"Best Random Forest parameters: {rf_grid.best_params_}")
    print(f"Best Gradient Boosting parameters: {gb_grid.best_params_}")
    
    # Save models
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open(svm_model_path, 'wb') as f:
        pickle.dump(svm_model, f)
        
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)
        
    with open(gb_model_path, 'wb') as f:
        pickle.dump(gb_model, f)
        
    with open(ensemble_model_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Return trained models and evaluation data
    models = {
        'logistic_regression': lr_model,
        'svm': svm_model,
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'ensemble': ensemble_model,
        'vectorizer': vectorizer,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw
    }
    
    if xgb_model:
        models['xgboost'] = xgb_model
    
    return models