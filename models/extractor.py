import spacy
from spacy.training.example import Example
import random
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class KeywordExtractor:
    def __init__(self, model_output_path="../models/mental_health_ner_model"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_output_path = model_output_path
        self.nlp = spacy.blank("en") 

        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")

    def fit(self, x_train, y_train, x_val, y_val, n_iter=10, batch_size=2):
        """Trains the NER model on the training set and evaluates on the validation set."""
        for extracted_concern in y_train:
            self.ner.add_label("CONCERN")
        
        training_examples = [
            Example.from_dict(
                self.nlp.make_doc(user_input),
                {"entities": [(user_input.find(concern), user_input.find(concern) + len(concern), "CONCERN")]}
            )
            for user_input, concern in zip(x_train, y_train)
        ]

        optimizer = self.nlp.initialize()

        for itn in range(n_iter):
            random.shuffle(training_examples)
            losses = {}
            
            for batch in tqdm(spacy.util.minibatch(training_examples, size=batch_size), desc=f"Iteration {itn + 1}/{n_iter}"):
                self.nlp.update(batch, drop=0.35, losses=losses)
            print(f"Losses at iteration {itn}: {losses}")
        
        self.nlp.to_disk(self.model_output_path)

        val_accuracy = self.evaluate_accuracy(x_val, y_val)
        print(f"Validation Accuracy: {val_accuracy:.2%}")

    def load_model(self):
        """Loads the trained model."""
        self.nlp = spacy.load(self.model_output_path)

    def predict(self, text):
        """Predicts entities in the given text."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def evaluate_accuracy(self, x, y):
        """Evaluates model accuracy on a given dataset (validation or test)."""
        y_pred = []

        for text, true_concern in zip(x, y):
            predicted_entities = self.predict(text)
            predicted_concern = predicted_entities[0][0] if predicted_entities else ""
            y_pred.append(predicted_concern)

        accuracy = accuracy_score(y, y_pred)
        return accuracy
