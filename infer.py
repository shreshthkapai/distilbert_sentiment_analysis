"""
Inference pipeline for DistilBERT sentiment analysis
File: infer.py (improved version)
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variables to cache model and tokenizer
_model = None
_tokenizer = None

def load_trained_model(model_path="./model"):
    """Load saved model and tokenizer (cached)"""
    global _model, _tokenizer
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Please train the model first.")
    
    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print(f"Loading model from {model_path}...")
    
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    print("Model loaded successfully!")
    return _model, _tokenizer

def predict_sentiment(text, model, tokenizer, max_length=256):
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text string
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_length: Max sequence length
        
    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    # Convert to readable format
    label = "Positive" if predicted_class == 1 else "Negative"
    
    return label, confidence

def predict(text, model_path="./model", max_length=256):
    """
    Simple prediction function for new text
    
    Args:
        text: Input text string
        model_path: Path to saved model
        max_length: Max sequence length
        
    Returns:
        String: "positive" or "negative"
    """
    try:
        # Load model and tokenizer (cached)
        model, tokenizer = load_trained_model(model_path)
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        
        return "positive" if predicted_class == 1 else "negative"
    
    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Prediction error: {str(e)}"