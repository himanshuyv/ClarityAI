import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class PolarityFinder:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # Load pre-trained model and tokenizer for binary sentiment classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize accuracy storage
        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None

    def predict_polarity(self, text):
        """Predicts the polarity of a single text input."""
        # Tokenize the text input and move inputs to device
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        
        # Get the model's prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted score
        scores = outputs.logits.softmax(dim=1).tolist()[0]  # Convert to probabilities

        # Simple rule-based approach for three classes
        if scores[1] > 0.5:  # High confidence positive
            return "Positive"
        elif scores[0] > 0.5:  # High confidence negative
            return "Negative"
        else:
            return "Neutral"

    def evaluate_accuracy(self, x_data, y_data):
        """Evaluates accuracy on the provided dataset."""
        correct_predictions = 0
        total = len(x_data)
        
        for text, actual_polarity in zip(x_data, y_data):
            predicted_polarity = self.predict_polarity(text)
            if predicted_polarity == actual_polarity:
                correct_predictions += 1
        
        accuracy = correct_predictions / total
        return accuracy

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Fits the model by calculating and storing train, validation, and test accuracies."""
        self.train_accuracy = self.evaluate_accuracy(x_train, y_train)
        self.val_accuracy = self.evaluate_accuracy(x_val, y_val)
        self.test_accuracy = self.evaluate_accuracy(x_test, y_test)

        # Save the model
        model_save_path = './models/polarity_model'
        os.makedirs(model_save_path, exist_ok=True)  # Create directory if it doesn't exist
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

    def load_model(self, model_path='./polarity_model'):
        """Loads the model and tokenizer from the specified directory."""
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)  # Move model to GPU if available
        else:
            raise ValueError(f"Model path {model_path} does not exist.")

    def get_accuracy(self):
        """Returns a dictionary of accuracies for train, validation, and test sets."""
        return {
            "Train Accuracy": self.train_accuracy,
            "Validation Accuracy": self.val_accuracy,
            "Test Accuracy": self.test_accuracy,
        }

    def get_predictions(self, x_data):
        """Returns a list of predictions for the given dataset."""
        predictions = []
        for text in x_data:
            predicted_polarity = self.predict_polarity(text)
            predictions.append(predicted_polarity)
        return predictions
