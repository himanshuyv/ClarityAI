import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the CSV data
data = pd.read_csv('../data/even_new_mental.csv')
data = data.iloc[:400]  # Using a sample of 400 for testing

# Load pre-trained model and tokenizer for binary sentiment classification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function for predicting polarity
def predict_polarity(text):
    # Tokenize the text input and move inputs to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the predicted score
    scores = outputs.logits.softmax(dim=1).tolist()[0]  # Convert to probabilities

    # Simple rule-based approach for three classes
    if scores[1] > 0.6:  # High confidence positive
        return "Positive"
    elif scores[0] > 0.6:  # High confidence negative
        return "Negative"
    else:
        return "Neutral"

# Initialize counters for accuracy calculation
correct_predictions = 0

# Iterate over each row to predict and print the polarity
for index, row in data.iterrows():
    user_input = row['User Input']
    actual_polarity = row['Polarity']
    predicted_polarity = predict_polarity(user_input)
    
    # Print input and prediction
    print(f"User Input: {user_input}")
    print(f"Actual Polarity: {actual_polarity} | Predicted Polarity: {predicted_polarity}\n")

    # Check if the prediction is correct
    if predicted_polarity == actual_polarity:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(data)
print(f"\nAccuracy: {accuracy:.2%}")

