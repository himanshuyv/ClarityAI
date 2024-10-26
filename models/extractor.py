import spacy
from spacy.training.example import Example
import random
import pandas as pd

# # Sample dataset
# data = [
#     ("I am constantly worried these days.", {"entities": [(5, 23, "CONCERN")]}),
#     ("I’m trying, but I’m still constantly worried.", {"entities": [(20, 38, "CONCERN")]}),
#     ("I am worried about health these days.", {"entities": [(5, 23, "CONCERN")]}),
#     ("Every day I’m happy and excited.", {"entities": [(12, 27, "CONCERN")]}),
#     ("I feel happy and excited lately.", {"entities": [(7, 22, "CONCERN")]}),
#     ("Sometimes, I think I'm feeling very low.", {"entities": [(24, 34, "CONCERN")]}),
#     ("My mind feels like it’s can't sleep well.", {"entities": [(22, 37, "CONCERN")]}),
#     ("It’s a struggle, I’m constantly worried.", {"entities": [(17, 35, "CONCERN")]}),
#     ("Lately, I’ve been feeling very anxious.", {"entities": [(18, 32, "CONCERN")]}),
#     ("Sometimes, I think I'm feeling hopeful.", {"entities": [(24, 39, "CONCERN")]}),
#     ("I’m trying, but I’m still extremely stressed.", {"entities": [(20, 37, "CONCERN")]}),
#     ("I am feeling very low these days.", {"entities": [(13, 23, "CONCERN")]}),
#     ("I am feeling hopeful, and it’s affecting me.", {"entities": [(13, 28, "CONCERN")]}),
#     ("My mind feels like it’s worried about health.", {"entities": [(22, 40, "CONCERN")]}),
#     ("I am constantly worried, and it’s affecting me.", {"entities": [(5, 23, "CONCERN")]}),
#     ("Recently, I’ve noticed that I’m constantly worried.", {"entities": [(31, 49, "CONCERN")]}),
#     ("I am feeling much better these days.", {"entities": [(5, 22, "CONCERN")]}),
# ]

# Sample data in the specified CSV format
data = pd.read_csv("../data/mental_health_dataset.csv")
data = data[100:200]

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
n_iter = 20
for itn in range(n_iter):
    random.shuffle(training_data)
    losses = {}
    for batch in spacy.util.minibatch(training_data, size=2):
        nlp.update(batch, drop=0.35, losses=losses)
    print(f"Losses at iteration {itn}: {losses}")

# Save the trained model
nlp.to_disk("mental_health_ner_model")

# Test the model
test_text = "I'm think I'm feeling depressed."
doc = nlp(test_text)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
