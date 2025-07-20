"""
Main pipeline for DistilBERT sentiment analysis project
File: main.py
"""

import os
import argparse
from train import (
    load_imdb_data, 
    preprocess_data, 
    load_model, 
    setup_trainer,
    train_model,
    evaluate_model,
    save_model
)
# Remove app import since we'll run it separately

def train_pipeline(subset_size=None):
    """Complete training pipeline"""
    print("=== Starting Training Pipeline ===")
    
    # 1. Load dataset
    dataset = load_imdb_data(subset_size=subset_size)
    
    # 2. Preprocess data
    tokenized_dataset, tokenizer = preprocess_data(dataset)
    
    # 3. Load model
    model = load_model()
    
    # 4. Setup trainer
    trainer = setup_trainer(
        model, 
        tokenizer, 
        tokenized_dataset["train"], 
        tokenized_dataset["test"]
    )
    
    # 5. Train model
    train_model(trainer)
    
    # 6. Evaluate model
    results = evaluate_model(trainer)
    
    # 7. Save model
    save_model(trainer, tokenizer)
    
    print("=== Training Pipeline Completed ===")
    return results

def main():
    parser = argparse.ArgumentParser(description="DistilBERT Sentiment Analysis - Training Only")
    parser.add_argument("--subset", type=int, default=None,
                       help="Use subset of data for training (for testing)")
    
    args = parser.parse_args()
    
    # Check if model already exists
    if os.path.exists("./model") and os.path.exists("./model/config.json"):
        response = input("Model already exists. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training...")
            print("To run the app: python app.py")
            return
    
    # Train the model
    train_pipeline(subset_size=args.subset)
    
    print("\n🎉 Training completed!")
    print("To run the app: python app.py")

if __name__ == "__main__":
    main()