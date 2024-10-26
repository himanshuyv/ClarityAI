import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the CSV data
data = pd.read_csv('./data/mental_health_dataset.csv')
# data=data.sample(n=100)
data = data.iloc[:100]
#User Input,Polarity,Extracted Concern,Category,Intensity

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

# Apply the function to your dataset
data['Predicted Polarity'] = data['User Input'].apply(predict_polarity)

# Save the results to a new CSV file
data.to_csv('./predictions_with_polarity.csv', index=False)
print("Predictions saved to './predictions_with_polarity.csv'")
