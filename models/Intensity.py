from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import joblib
import os

class IntensityScorer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.data = []

    def calculate_intensity(self, compound):
        """Calculates intensity based on the compound score."""
        if compound >= 0:
            if compound > 0.65:
                return 4
            elif compound > 0.4:
                return 3
            elif compound > 0.3:
                return 2
            else:
                return 1
        elif compound < 0:
            if compound > -0.2:
                return 1
            elif compound > -0.3:
                return 2
            elif compound > -0.4:
                return 3
            elif compound > -0.8:
                return 4
            else:
                return 5

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Fits the model using the provided training data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        self.data = []
        for sentence in x_train:
            vader_scores = self.analyzer.polarity_scores(sentence)
            intensity = self.calculate_intensity(vader_scores['compound'])
            self.data.append(intensity)
        self.save_model('../models/Intensity_model')

    def predict(self,text):
        """Predicts the intensities for the test dataset."""
        # predictions = []
        # for sentence in self.x_test:
        vader_scores = self.analyzer.polarity_scores(text)
        intensity = self.calculate_intensity(vader_scores['compound'])
            # predictions.append(intensity)
        return intensity

    def evaluate_accuracy(self):
        """Evaluates the model accuracy on the test dataset."""
        predictions = self.predict()
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
    
    def get_predictions(self,text):
        """Returns the list of predictions for the test dataset."""
        return self.predict(text)

    def save_model(self, model_path='../models/Intensity_scorer_model'):
        """Save the trained model to the specified path."""
        joblib.dump(self, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path='../models/Intensity_scorer_model'):
        """Load the model from the specified path."""
        print(model_path)
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            # self.__dict__.update(loaded_model.__dict__)
            print(f"Model loaded from {model_path}")
        else:
            raise ValueError(f"Model path {model_path} does not exist.")
