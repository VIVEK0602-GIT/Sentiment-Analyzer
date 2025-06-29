from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from tqdm import tqdm

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model with comprehensive metrics
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : sparse matrix
        Test feature matrix
    y_test : array-like
        Test labels
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_prob = None
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print model performance
    print(f"\n{model_name} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\n" + classification_report(y_test, y_pred))
    
    # Cross-validation if the model supports it
    try:
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_weighted')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
    except Exception as e:
        print(f"Cross-validation not available: {e}")
        cv_scores = None
    
    # Return metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cv_scores': cv_scores
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, model_name, normalized=True, save_path=None):
    """
    Plot confusion matrix with enhanced visualization
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list
        List of class names
    model_name : str
        Name of the model
    normalized : bool
        Whether to normalize the confusion matrix
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        title = f'Confusion Matrix - {model_name}'
    
    # Create figure with custom figsize
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with better coloring and annotations
    ax = sns.heatmap(cm, annot=True, fmt='.2f' if normalized else 'd', 
                    cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    # Set plot properties
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_confusion_matrix.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_test, y_prob, classes, model_name, save_path=None):
    """
    Plot ROC curve for multiclass classification
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    classes : list
        List of class names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    if y_prob is None:
        print("Probability predictions not available for ROC curve")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Convert class labels to one-hot encoding for ROC curve
    y_test_bin = pd.get_dummies(y_test)
    
    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin.iloc[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin.iloc[:, i], y_prob[:, i])
        
        # Plot ROC curve for each class
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve of class {cls} (area = {roc_auc[i]:.2f})')
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right")
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_roc_curve.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")

def plot_precision_recall_curve(y_test, y_prob, classes, model_name, save_path=None):
    """
    Plot precision-recall curve for multiclass classification
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    classes : list
        List of class names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    if y_prob is None:
        print("Probability predictions not available for precision-recall curve")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Convert class labels to one-hot encoding
    y_test_bin = pd.get_dummies(y_test)
    
    # Calculate precision-recall curve for each class
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin.iloc[:, i], y_prob[:, i])
        avg_precision = average_precision_score(y_test_bin.iloc[:, i], y_prob[:, i])
        
        # Plot precision-recall curve for each class
        plt.plot(recall, precision, lw=2,
                 label=f'{cls} (AP = {avg_precision:.2f})')
    
    # Set plot properties
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16)
    plt.legend(loc="best")
    plt.grid(True)
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_precision_recall_curve.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-recall curve saved to {save_path}")

def analyze_misclassifications(X_raw, y_true, y_pred, classes, model_name, save_path=None):
    """
    Analyze misclassified instances
    
    Parameters:
    -----------
    X_raw : array-like
        Raw text data
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list
        List of class names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the analysis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame of misclassified instances
    """
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'text': X_raw,
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': y_true == y_pred
    })
    
    # Filter misclassified instances
    misclassified = results_df[~results_df['correct']]
    
    # Group by true/predicted label combinations
    error_counts = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
    
    # Sort by count
    error_counts = error_counts.sort_values('count', ascending=False)
    
    # Print top misclassifications
    print(f"\nTop misclassifications for {model_name}:")
    print(error_counts.head(10))
    
    # Save detailed analysis if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_misclassifications.csv'
    
    misclassified.to_csv(save_path, index=False)
    print(f"Misclassification analysis saved to {save_path}")
    
    return misclassified

def save_metrics_to_csv(metrics_list, save_path=None):
    """
    Save model metrics to CSV
    
    Parameters:
    -----------
    metrics_list : list
        List of metrics dictionaries
    save_path : str, optional
        Path to save the CSV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame of metrics
    """
    # Extract metrics to DataFrame
    metrics_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1_score'],
        }
        for m in metrics_list
    ])
    
    # Save to CSV
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = 'results/model_metrics.csv'
    
    metrics_df.to_csv(save_path, index=False)
    print(f"Model metrics saved to {save_path}")
    
    return metrics_df

def plot_metrics_comparison(metrics_df, save_path=None):
    """
    Create interactive comparison plots of model metrics using Plotly
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame of model metrics
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Create bar chart with Plotly
    fig = px.bar(
        metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metric',
        font=dict(size=14),
        yaxis=dict(range=[0, 1])
    )
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = 'results/model_comparison.png'
    
    fig.write_image(save_path, width=1000, height=600)
    fig.write_html('results/model_comparison.html')
    
    print(f"Model comparison plot saved to {save_path}")

def plot_feature_importance(model, feature_names, model_name, n_top=20, save_path=None):
    """
    Plot feature importance for models that support it
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : array-like
        Feature names
    model_name : str
        Name of the model
    n_top : int
        Number of top features to plot
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Check if model supports feature importance
    if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not support feature importance extraction")
        return
    
    # Extract feature importance
    if hasattr(model, 'coef_'):
        # For linear models
        if len(model.coef_.shape) > 1:
            # For multiclass
            importance = np.sum(np.abs(model.coef_), axis=0)
        else:
            importance = np.abs(model.coef_)
    else:
        # For tree-based models
        importance = model.feature_importances_
    
    # Create DataFrame of feature importance
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance and take top n
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(n_top)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot horizontal bar chart
    ax = sns.barplot(x='importance', y='feature', data=feat_imp, color='steelblue')
    
    # Set plot properties
    plt.title(f'Top {n_top} Feature Importance - {model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_feature_importance.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {save_path}")

def create_interactive_dashboard(metrics_list, save_path=None):
    """
    Create an interactive HTML dashboard with all evaluation metrics
    
    Parameters:
    -----------
    metrics_list : list
        List of metrics dictionaries
    save_path : str, optional
        Path to save the dashboard
        
    Returns:
    --------
    None
    """
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1_score'],
        }
        for m in metrics_list
    ])
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Accuracy', 'Model Precision',
            'Model Recall', 'Model F1 Score'
        )
    )
    
    # Add bar charts for each metric
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['Accuracy'], name='Accuracy'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['Precision'], name='Precision'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['Recall'], name='Recall'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['F1 Score'], name='F1 Score'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Model Performance Dashboard',
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    # Add annotations with exact values
    for i, col in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
        row = i // 2 + 1
        col_idx = i % 2 + 1
        
        for j, model in enumerate(metrics_df['Model']):
            value = metrics_df.loc[metrics_df['Model'] == model, col].values[0]
            
            fig.add_annotation(
                x=model,
                y=value,
                text=f"{value:.3f}",
                showarrow=True,
                arrowhead=1,
                row=row,
                col=col_idx
            )
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = 'results/model_dashboard.html'
    
    fig.write_html(save_path)
    
    print(f"Interactive dashboard saved to {save_path}")
    
    # Return a static image version as well
    static_path = 'results/model_dashboard.png'
    fig.write_image(static_path, width=1200, height=800)
    
    print(f"Static dashboard image saved to {static_path}")

# Additional functions for analyzing specific aspects of models
def visualize_sentiment_distribution(y_true, y_pred, model_name, save_path=None):
    """
    Visualize the distribution of true vs predicted sentiment
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Create DataFrame for analysis
    pred_df = pd.DataFrame({
        'True Sentiment': y_true,
        'Predicted Sentiment': y_pred
    })
    
    # Count true and predicted sentiment distributions
    true_counts = pred_df['True Sentiment'].value_counts().reset_index()
    true_counts.columns = ['Sentiment', 'Count']
    true_counts['Type'] = 'True'
    
    pred_counts = pred_df['Predicted Sentiment'].value_counts().reset_index()
    pred_counts.columns = ['Sentiment', 'Count']
    pred_counts['Type'] = 'Predicted'
    
    # Combine for plotting
    combined_counts = pd.concat([true_counts, pred_counts])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    sns.barplot(x='Sentiment', y='Count', hue='Type', data=combined_counts)
    
    # Set plot properties
    plt.title(f'True vs Predicted Sentiment Distribution - {model_name}', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Source')
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_sentiment_distribution.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sentiment distribution plot saved to {save_path}")

def save_evaluation_results(metrics_list, output_dir='results', prefix=''):
    """
    Save comprehensive evaluation results (metrics, plots, etc.)
    
    Parameters:
    -----------
    metrics_list : list
        List of dictionaries containing model metrics
    output_dir : str
        Directory to save the results
    prefix : str, optional
        Prefix for saved files
    
    Returns:
    --------
    dict
        Dictionary with paths to saved files
    """
    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{prefix}_{timestamp}" if prefix else timestamp
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, f'{prefix}_metrics.csv')
    metrics_df = save_metrics_to_csv(metrics_list, save_path=metrics_path)
    
    # Plot and save metrics comparison
    comparison_path = os.path.join(output_dir, f'{prefix}_model_comparison.png')
    plot_metrics_comparison(metrics_df, save_path=comparison_path)
    
    # Save interactive metrics comparison
    interactive_comparison_path = os.path.join(output_dir, f'{prefix}_interactive_model_comparison.html')
    plot_interactive_metrics_comparison(metrics_df, save_path=interactive_comparison_path)
    
    # Save confusion matrices and other plots for each model
    model_results = {}
    for metrics in metrics_list:
        model_name = metrics['model_name']
        y_test = metrics['y_test'] if 'y_test' in metrics else None
        y_pred = metrics['y_pred']
        
        # Clean model name for file naming
        clean_name = model_name.replace(' ', '_').lower()
        
        # Get class names
        class_names = metrics.get('class_names', [str(i) for i in range(metrics['num_classes'])])
        
        # Save confusion matrix
        if y_test is not None:
            cm_path = os.path.join(output_dir, f'{prefix}_{clean_name}_confusion_matrix.png')
            plot_confusion_matrix(y_test, y_pred, class_names, model_name, save_path=cm_path)
            
            # Save interactive confusion matrix
            interactive_cm_path = os.path.join(output_dir, f'{prefix}_{clean_name}_interactive_confusion_matrix.html')
            plot_interactive_confusion_matrix(y_test, y_pred, class_names, model_name, save_path=interactive_cm_path)
        
        # Save ROC curve for binary classification
        if 'y_proba' in metrics and 'roc_auc' in metrics:
            roc_path = os.path.join(output_dir, f'{prefix}_{clean_name}_roc_curve.png')
            plot_roc_curve(y_test, metrics['y_proba'], class_names, model_name, save_path=roc_path)
            
            # Save PR curve
            pr_path = os.path.join(output_dir, f'{prefix}_{clean_name}_pr_curve.png')
            plot_precision_recall_curve(y_test, metrics['y_proba'], class_names, model_name, save_path=pr_path)
        
        # Save classification report plot
        if 'classification_report' in metrics:
            report_path = os.path.join(output_dir, f'{prefix}_{clean_name}_classification_report.png')
            plot_classification_report(metrics['classification_report'], model_name, save_path=report_path)
        
        # Add to model results
        model_results[model_name] = {
            'metrics': {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_proba', 'y_test']}
        }
    
    # Save all results as JSON
    results_summary = {
        'timestamp': timestamp,
        'models': model_results
    }
    
    json_path = os.path.join(output_dir, f'{prefix}_evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, default=str)
    
    return {
        'metrics_csv': metrics_path,
        'comparison_plot': comparison_path,
        'interactive_comparison': interactive_comparison_path,
        'summary_json': json_path,
        'model_results': model_results
    } 