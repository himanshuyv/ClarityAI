import pandas as pd
from sklearn.model_selection import train_test_split

from polarity import PolarityFinder
from extractor import KeywordExtractor
from categorizer import CategoryClassifier

data = pd.read_csv('../data/synthetic.csv').sample(frac=1, random_state=42)
x_user = data['User Input']
x_concern = data['Extracted Concern']
y_polarity = data['Polarity']
y_concern = data['Extracted Concern']
y_category = data['Category']

# x_train_polarity, x_temp_polarity, y_train_polarity, y_temp_polarity = train_test_split(x_user, y_polarity, test_size=0.3, random_state=42)  # 70% train, 30% temp
# x_val_polarity, x_test_polarity, y_val_polarity, y_test_polarity = train_test_split(x_temp_polarity, y_temp_polarity, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
# polarity_model = PolarityFinder()
# polarity_model.fit(x_train_polarity, y_train_polarity, x_val_polarity, y_val_polarity, x_test_polarity, y_test_polarity)
# polarity_accuracy = polarity_model.get_accuracy()
# print("Polarity Accuracy Results:", polarity_accuracy)

# x_train_extractor, x_temp_extractor, y_train_extractor, y_temp_extractor = train_test_split(x_user, y_concern, test_size=0.3, random_state=42)  # 70% train, 30% temp
# x_val_extractor, x_test_extractor, y_val_extractor, y_test_extractor = train_test_split(x_temp_extractor, y_temp_extractor, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
# ner_model = KeywordExtractor()
# ner_model.fit(x_train_extractor, y_train_extractor, x_val_extractor, y_val_extractor, n_iter=10)
# ner_model.load_model()
# test_accuracy = ner_model.evaluate_accuracy(x_test_extractor, y_test_extractor)
# print(f"Extractor Test Accuracy: {test_accuracy:.2%}")

x_train_categorizer, x_temp_categorizer, y_train_categorizer, y_temp_categorizer = train_test_split(
    x_concern, y_category, test_size=0.3, random_state=42)  # 70% train, 30% temp
x_val_categorizer, x_test_categorizer, y_val_categorizer, y_test_categorizer = train_test_split(
    x_temp_categorizer, y_temp_categorizer, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
classifier = CategoryClassifier()
classifier.fit(x_train_categorizer, y_train_categorizer)
classifier.report_performance(x_test_categorizer, y_test_categorizer)