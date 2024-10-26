import spacy
from spacy.training.example import Example
import random
import pandas as pd

# Sample data in the specified CSV format
data = pd.read_csv("../data/mental_health_dataset.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the list of tuples
data = []

for index, row in df.iterrows():
    user_input = row["User Input"]
    extracted_concern = row["Extracted Concern"]
    
    start_index = user_input.find(extracted_concern)
    end_index = start_index + len(extracted_concern)
    
    data.append((user_input, {"entities": [(start_index, end_index, "CONCERN")]}))

# Initialize a blank English model
nlp = spacy.blank("en")

# Add NER component if it's not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER component
for _, annotations in data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Prepare the data for training
def prepare_training_data(data):
    training_data = []
    for text, annotations in data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        training_data.append(example)
    return training_data

# Convert data to spaCy's training format
training_data = prepare_training_data(data)

# Training the NER model
optimizer = nlp.initialize()

# Train the model
n_iter = 15
for itn in range(n_iter):
    random.shuffle(training_data)
    losses = {}
    for batch in spacy.util.minibatch(training_data, size=2):
        nlp.update(batch, drop=0.35, losses=losses)
    print(f"Losses at iteration {itn}: {losses}")

model_output_path = "mental_health_ner_model"
nlp.to_disk(model_output_path)
print(f"Model saved to: {model_output_path}")

# Function to load the trained model
def load_model(model_path):
    """Load a trained spaCy NER model from the specified path."""
    return spacy.load(model_path)

# Example usage of the load_model function
loaded_nlp = load_model(model_output_path)

# Test the loaded model
test_text = "I am feeling Elon Muskish today."
doc = loaded_nlp(test_text)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
