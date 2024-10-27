import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from models.polarity import PolarityFinder
from models.extractor import KeywordExtractor
from models.Intensity import IntensityScorer
from models.categorizer import CategoryClassifier

def model_inference(fetched_input):
    data = pd.read_csv('../data/new_synth2.csv').sample(frac=1, random_state=42)
    data.to_csv('../data/new_synth2.csv', index=False)
    x_user = data['User Input']
    x_concern = data['Extracted Concern']
    y_polarity = data['Polarity']
    y_concern = data['Extracted Concern']
    y_category = data['Category']
    y_intensity = data['Intensity']
    fetched_sentence= 'The cat running around means I have trouble sleeping sometimes.'
    fetched_sentence= fetched_input
    # x_train_polarity, x_temp_polarity, y_train_polarity, y_temp_polarity = train_test_split(x_user, y_polarity, test_size=0.3, random_state=42)  # 70% train, 30% temp
    # x_val_polarity, x_test_polarity, y_val_polarity, y_test_polarity = train_test_split(x_temp_polarity, y_temp_polarity, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
    # polarity_model = PolarityFinder()
    # polarity_model.fit(x_train_polarity, y_train_polarity, x_val_polarity, y_val_polarity, x_test_polarity, y_test_polarity)
    # polarity_accuracy = polarity_model.get_accuracy()
    # print("Polarity Accuracy Results:", polarity_accuracy)
    polarity_model = PolarityFinder()
    polarity_model.load_model('../models/polarity_model')
    print('polarity:',polarity_model.get_predictions([fetched_sentence]))
    polarity_prediction = polarity_model.get_predictions([fetched_sentence])

    # x_train_extractor, x_temp_extractor, y_train_extractor, y_temp_extractor = train_test_split(x_user, y_concern, test_size=0.3, random_state=42)  # 70% train, 30% temp
    # x_val_extractor, x_test_extractor, y_val_extractor, y_test_extractor = train_test_split(x_temp_extractor, y_temp_extractor, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
    # ner_model = KeywordExtractor()
    # ner_model.fit(x_train_extractor, y_train_extractor, x_val_extractor, y_val_extractor, n_iter=10)
    ner_model = KeywordExtractor()
    ner_model.load_model()
    print('extracted phrase:',ner_model.predict(fetched_sentence))
    
    extracted_phrase = ner_model.predict(fetched_sentence)
    if len(extracted_phrase)>0:
        extracted_phrase=extracted_phrase[0][0]
    else:
        return [polarity_prediction,'abort','abort','abort']
    # test_accuracy = ner_model.evaluate_accuracy(x_test_extractor, y_test_extractor)
    # print(f"Extractor Test Accuracy: {test_accuracy:.2%}")

    # x_train_categorizer, x_temp_categorizer, y_train_categorizer, y_temp_categorizer = train_test_split(
    #     x_concern, y_category, test_size=0.3, random_state=42)  # 70% train, 30% temp
    # x_val_categorizer, x_test_categorizer, y_val_categorizer, y_test_categorizer = train_test_split(
    #     x_temp_categorizer, y_temp_categorizer, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
    # classifier = CategoryClassifier()
    # classifier.fit(x_train_categorizer, y_train_categorizer)
    classifier= CategoryClassifier()
    classifier.load_model('../models/categorizer_model')
    print('Category:',classifier.predict_single(extracted_phrase))
    category_prediction = classifier.predict_single(extracted_phrase)
    # classifier.report_performance(x_test_categorizer, y_test_categorizer)

    # x_train_intensity, x_temp_intensity, y_train_intensity, y_temp_intensity = train_test_split(x_concern, y_intensity, test_size=0.3, random_state=42)  # 70% train, 30% temp
    # x_val_intensity, x_test_intensity, y_val_intensity, y_test_intensity = train_test_split(x_temp_intensity, y_temp_intensity, test_size=0.5, random_state=42)  # Split 30% into 15% val and 15% test
    # vader_analyzer = IntensityScorer()
    # vader_analyzer.fit(x_train_intensity, y_train_intensity, x_val_intensity, y_val_intensity, x_test_intensity, y_test_intensity)
    vader_analyzer = IntensityScorer()
    vader_analyzer.load_model('../models/Intensity_model')
    print('intensity:',vader_analyzer.get_predictions(extracted_phrase))
    intensity_prediction = vader_analyzer.get_predictions(extracted_phrase)
    # intensity_accuracy = vader_analyzer.evaluate_accuracy()
    # print(f"Intensity Accuracy: {intensity_accuracy:.2%}")

    # categories = classifier.predict_list(x_test_categorizer)
    # intensities = vader_analyzer.get_predictions()
    # polarities = polarity_model.get_predictions(x_test_polarity)

    # trends = []
    # num_predicitions = len(categories)
    # for i in range(num_predicitions):
    #     num = 0
    #     if polarities[i] == "Positive":
    #         num = 1
    #     else:
    #         num = -1
    #     trends.append([num * intensities[i], categories[i]])
    # print(trends)
    return [polarity_prediction,extracted_phrase,category_prediction,intensity_prediction]
