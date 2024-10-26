import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

def split_data(x, y, train_size, val_size):
    train_size /= 100.0
    val_size /= 100.0
    n = x.shape[0]
    train_end = int(train_size * n)
    val_end = int((train_size + val_size) * n)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    return x_train, y_train, x_val, y_val, x_test, y_test

# Step 1: Load the dataset
data = pd.read_csv("../data/even_new_mental.csv")  # Adjust the path to your dataset

# Filter necessary columns
data = data[['Extracted Concern', 'Category']]

# Step 2: Prepare the dataset for training
X = data['Extracted Concern']
y = data['Category']

# Split the dataset into training and testing sets (80:20 ratio)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 80, 10)

# Step 3: Create a pipeline for TF-IDF vectorization and classifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 4: Function to test the model with new phrases
def predict_category(phrase):
    """Predict the category for a given phrase using the trained model."""
    return model.predict([phrase])[0]

# Example test
test_phrases = X_test

# Display predictions
for phrase in test_phrases:
    category = predict_category(phrase)
    print(f"Phrase: '{phrase}' -> Predicted Category: '{category}'")
