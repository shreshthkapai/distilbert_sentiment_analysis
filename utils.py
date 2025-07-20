"""
Utility functions for DistilBERT sentiment analysis
File: utils.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for binary classification
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dict with accuracy, f1, precision, recall
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }