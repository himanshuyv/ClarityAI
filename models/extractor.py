import spacy
from spacy.training.example import Example
import random
import pandas as pd
import torch
from tqdm import tqdm

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Sample data in the specified CSV format
data = pd.read_csv("../data/even_new_mental.csv")
# data = data.sample(n=50000)

data = data.sample(frac=1)

test_data = data[400:]
data = data[:400]

# Convert to DataFrame
df = pd.DataFrame(data)

# # Create the list of tuples
training_data = []

for index, row in df.iterrows():
    user_input = row["User Input"]
    extracted_concern = row["Extracted Concern"]
    
    start_index = user_input.find(extracted_concern)
    end_index = start_index + len(extracted_concern)
    
    training_data.append((user_input, {"entities": [(start_index, end_index, "CONCERN")]}))

# Initialize a blank English model
nlp = spacy.blank("en")

# Add NER component if it's not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER component
for _, annotations in training_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Prepare the data for training
def prepare_training_data(data):
    training_examples = []
    for text, annotations in data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        training_examples.append(example)
    return training_examples

# Convert data to spaCy's training format
training_examples = prepare_training_data(training_data)

# Training the NER model``
optimizer = nlp.initialize()

# Train the model
n_iter = 10
for itn in range(n_iter):
    random.shuffle(training_examples)
    losses = {}
    
    # Create minibatches and wrap in tqdm for progress bar
    for batch in tqdm(spacy.util.minibatch(training_examples, size=2), desc=f"Iteration {itn + 1}/{n_iter}"):
        nlp.update(batch, drop=0.35, losses=losses)
    
    print(f"Losses at iteration {itn}: {losses}")

# # Save the model
model_output_path = "even_new_mental_health_ner_model"
nlp.to_disk(model_output_path)

# Function to load the trained model
def load_model(model_path):
    """Load a trained spaCy NER model from the specified path."""
    return spacy.load(model_path)

# Example usage of the load_model function
loaded_nlp = load_model(model_output_path)

# test_texts = test_data["User Input"].tolist()

test_texts = [
    "I've been feeling very anxious lately.",
    "I'm so tired all the time and can't seem to focus on anything.",
    "Lately I've been feeling really depressed and hopeless.",
    "I get nervous around new people",
    "I'm constantly worried about everything.",
    "I’m trying, but I’m still feeling very anxious.",
    "I feel hopeful sometimes and sometimes im lonely and confused",
    "I am feeling hurt and depression",
    "Im so sad",
]

# Process each test text
for text in test_texts:
    doc = loaded_nlp(text)
    print(f"\nInput: {text}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])