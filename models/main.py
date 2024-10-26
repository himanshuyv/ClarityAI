import pandas as pd
from sklearn.model_selection import train_test_split

from polarity import PolarityFinder
data = pd.read_csv('../data/synthetic.csv').sample(frac=1, random_state=42)
x = data['User Input']
y = data['Polarity']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test

polarity_model = PolarityFinder()
polarity_model.fit(x_train, y_train, x_val, y_val, x_test, y_test)

polarity_accuracy = polarity_model.get_accuracy()
print("Accuracy Results:", polarity_accuracy)