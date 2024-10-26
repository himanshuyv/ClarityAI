from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

class IntensityScorer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

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

    def predict(self):
        """Predicts the intensities for the test dataset."""
        predictions = []
        for sentence in self.x_test:
            vader_scores = self.analyzer.polarity_scores(sentence)
            intensity = self.calculate_intensity(vader_scores['compound'])
            predictions.append(intensity)
        return predictions

    def evaluate_accuracy(self):
        """Evaluates the model accuracy on the test dataset."""
        predictions = self.predict()
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
