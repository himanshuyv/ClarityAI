import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming you have a DataFrame `df` with 'sentence' and 'intensity' columns

# Load data
data = {
    "sentence": [
        "worried about health",
        "constantly worried",
        "feeling a bit low",
        "extremely anxious",
        "slightly concerned",
        # Add more sentences with labels
    ],
    "intensity": [7, 6, 4, 9, 3]
}
df = pd.DataFrame(data)

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2)

# Define Dataset class
class IntensityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        intensity = self.data.iloc[idx]['intensity']
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(intensity, dtype=torch.float)
        }

# Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Dataset and DataLoader
train_dataset = IntensityDataset(train_df, tokenizer, max_len=32)
val_dataset = IntensityDataset(val_df, tokenizer, max_len=32)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=300,  # Set a high number of epochs but rely on early stopping
    weight_decay=0.01,
    logging_dir='./logs',
)

# Custom Callback for Early Stopping on Validation Loss
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=0.002):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Check validation loss and stop training if it goes below threshold
        if metrics and 'eval_loss' in metrics and metrics['eval_loss'] < self.threshold:
            print(f"Stopping training early as validation loss has reached {metrics['eval_loss']:.4f}")
            control.should_training_stop = True

# Instantiate Trainer with Early Stopping Callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    callbacks=[EarlyStoppingCallback(threshold=0.002)],
)

# Train the model
trainer.train()

# Prediction function
def predict_intensity(sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the device

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=32)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.cpu().item()  # Move output to CPU before fetching the scalar

    return min(max(0, score), 10)  # Clamp between 0 and 10

# Example usage
print(predict_intensity("extremely anxious"))
