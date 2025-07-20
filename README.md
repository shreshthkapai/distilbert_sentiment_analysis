# Movie Review Sentiment Analysis with DistilBERT

This project provides a complete pipeline for fine-tuning a DistilBERT model for sentiment analysis on movie reviews, along with a Gradio-based web interface for interactive predictions.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shreshth2002/distilbert-sentiment)

## Features

*   **Fine-tuned DistilBERT:** Utilizes a pre-trained DistilBERT model, fine-tuned on the IMDB dataset for high accuracy.
*   **Training Pipeline:** Includes scripts for a complete training pipeline, from data loading and preprocessing to model training and evaluation.
*   **Gradio Interface:** An intuitive web interface built with Gradio for easy, interactive sentiment prediction.
*   **Modular Code:** The project is organized into clear and understandable modules for training, inference, and the user interface.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://huggingface.co/spaces/Shreshth2002/distilbert-sentiment
    cd distilbert-sentiment
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model from scratch, run the `main.py` script:

```bash
python main.py
```

This will download the IMDB dataset, preprocess it, fine-tune the DistilBERT model, and save the trained model and tokenizer to the `./model` directory.

You can also train on a smaller subset of the data for faster testing:

```bash
python main.py --subset 1000
```

### Running the App

Once the model is trained, you can launch the Gradio web interface:

```bash
python app.py
```

This will start a local web server, and you can access the app in your browser at the provided URL (usually `http://127.0.0.1:7860`).

## Project Structure

```
.
├── app.py              # Gradio application
├── main.py             # Main training pipeline
├── train.py            # Training and evaluation logic
├── infer.py            # Inference functions
├── utils.py            # Utility functions (e.g., metrics)
├── requirements.txt    # Project dependencies
├── model/              # Saved model and tokenizer
└── README.md           # This file
```

## Technical Deep Dive

This section details the technical decisions and architecture of the training pipeline.

### Model Architecture

The core of this project is the `distilbert-base-uncased` model from Hugging Face. This model was chosen for its balance of performance and computational efficiency. It is a distilled version of BERT, approximately 40% smaller, while retaining 97% of its language understanding capabilities, making it ideal for deployment and faster inference. The model is configured for sequence classification with two labels (positive/negative).

### Data Preprocessing and Tokenization

The IMDB dataset is preprocessed using the `distilbert-base-uncased` tokenizer. The following steps are applied to each review:

1.  **Tokenization**: Texts are converted into tokens that correspond to the model's vocabulary.
2.  **Padding**: Sequences are padded to a uniform length of `max_length=256`. This ensures that all input tensors have the same shape.
3.  **Truncation**: Sequences longer than 256 tokens are truncated to fit the model's input size.
4.  **Attention Masks**: Attention masks are generated to differentiate between tokens and padding.

This entire process is mapped across the dataset in a batched fashion for efficiency.

### Training Configuration

The model is trained using the Hugging Face `Trainer` API, which abstracts away the training loop. The training is configured with the following key hyperparameters defined in `TrainingArguments`:

*   **Batch Size**: `per_device_train_batch_size=2` and `per_device_eval_batch_size=4`.
*   **Gradient Accumulation**: `gradient_accumulation_steps=2` is used to simulate a larger batch size (effective train batch size of 4) without increasing memory consumption. This helps in stabilizing training.
*   **Epochs**: The model is trained for `num_train_epochs=3`.
*   **Evaluation Strategy**: The model is evaluated at the end of each epoch (`eval_strategy="epoch"`).
*   **Best Model Selection**: The best model checkpoint is saved based on the F1 score on the evaluation set (`metric_for_best_model="f1"`, `load_best_model_at_end=True`). The F1 score is chosen as it provides a balanced measure of precision and recall.
*   **Optimizer**: The `Trainer` defaults to the AdamW optimizer, which is well-suited for training Transformer models.

### Evaluation Metrics

The model's performance is evaluated using a suite of metrics computed on the test set:

*   **Accuracy**: The proportion of correctly classified reviews.
*   **F1 Score**: The harmonic mean of precision and recall, providing a single score that balances both concerns.
*   **Precision**: The ratio of true positives to the total number of predicted positives.
*   **Recall**: The ratio of true positives to the total number of actual positives.

These metrics are calculated using `scikit-learn` within the `compute_metrics` function, which is passed to the `Trainer`.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
