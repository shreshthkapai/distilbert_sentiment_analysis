"""
Training and evaluation logic for DistilBERT sentiment analysis
File: train.py
"""

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    logging
)

# Local imports
from utils import compute_metrics
from datasets import load_dataset

# Standard library imports
import torch
import numpy as np
import pandas as pd

# Sklearn metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

# Suppress HF log spam
logging.set_verbosity_error()

# ===== DATASET LOADING =====

def load_imdb_data(subset_size=None):
    """Load IMDB dataset with optional subsampling"""
    dataset = load_dataset("imdb")
    
    # Optional subsetting for memory constraints
    if subset_size:
        dataset["train"] = dataset["train"].select(range(subset_size))
        dataset["test"] = dataset["test"].select(range(min(subset_size // 4, len(dataset["test"]))))
    
    print(f"Dataset loaded - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    return dataset

# ===== PREPROCESSING =====

def preprocess_data(dataset, max_length=256):
    """Tokenize and prepare dataset for training"""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize both splits
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Rename label column and set format
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset, tokenizer

# ===== MODEL LOADING =====

def load_model():
    """Load pre-trained DistilBERT model for sequence classification"""
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        return_dict=True
    )
    return model

# ===== TRAINING SETUP =====

def get_training_args():
    """Define training arguments"""
    return TrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42
    )

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """Initialize Trainer with model and datasets"""
    training_args = get_training_args()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    return trainer

# ===== TRAIN & EVALUATE =====

def train_model(trainer):
    """Train the model"""
    print("Starting training...")
    trainer.train()
    print("Training completed!")

def evaluate_model(trainer):
    """Evaluate the trained model"""
    print("Evaluating model...")
    results = trainer.evaluate()
    
    print("=== Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    return results

# ===== SAVE MODEL =====

def save_model(trainer, tokenizer, save_path="./model"):
    """Save trained model and tokenizer"""
    print(f"Saving model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")