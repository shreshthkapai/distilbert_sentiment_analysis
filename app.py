"""
Gradio frontend for DistilBERT sentiment analysis
File: app.py
"""

import gradio as gr
from infer import predict

def sentiment_analyzer(text):
    """Wrapper function for Gradio interface"""
    if not text.strip():
        return "Please enter some text"
    
    result = predict(text)
    return result.capitalize()

# Create Gradio interface
interface = gr.Interface(
    fn=sentiment_analyzer,
    inputs=gr.Textbox(
        label="Movie Review",
        placeholder="Enter your movie review here...",
        lines=3
    ),
    outputs=gr.Label(label="Sentiment Prediction"),
    title="🎬 Movie Review Sentiment Analysis",
    description="Fine-tuned DistilBERT model for movie review sentiment classification",
    examples=[
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible film, worst movie I've ever seen. Complete waste of time.",
        "The movie was okay, not great but not terrible either."
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)