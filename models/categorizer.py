from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score

class CategoryClassifier:
    def __init__(self, model=None):
        self.model = model or make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

    def fit(self, x_train, y_train):
        """Train the model with training data."""
        self.model.fit(x_train, y_train)

    def predict(self, x):
        """Predict categories for given input data."""
        return self.model.predict(x)

    def predict_list(self, phrases):
        """Predict categories for a list of phrases."""
        return self.predict(phrases)

    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data, returning accuracy and precision."""
        y_pred = self.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        return accuracy, precision

    def predict_single(self, phrase):
        """Predict the category for a single phrase."""
        return self.model.predict([phrase])[0]

    def report_performance(self, x_test, y_test):
        """Prints accuracy and precision on test data."""
        accuracy, precision = self.evaluate(x_test, y_test)
        print(f"Classifier Accuracy: {accuracy:.2%}")
        print(f"Classifier Precision: {precision:.2%}")