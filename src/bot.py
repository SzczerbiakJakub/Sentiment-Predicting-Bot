import tensorflow as tf
from tensorflow import keras
import numpy as np



positive_test = ["Just finished an amazing workout! Feeling energized and ready to take on the day 💪 #fitness #positivevibes",
        "Spent the afternoon catching up with old friends. So grateful for the laughter and memories shared! #friendship #gratitude",
        "Reached a major milestone at work today! Hard work and perseverance really do pay off. Excited for what the future holds! #success #career",
        "Enjoying a beautiful sunset with loved ones. Sometimes it's the simple moments that bring the most joy. #familytime #grateful",
        "Received some great news today! Feeling incredibly blessed and thankful for the opportunities that come my way. #blessed #gratitude"
                 ]

negative_test = np.array([["Feeling so frustrated with this never-ending traffic jam. Can't believe I'm going to be late for work again! 😡 #trafficwoes"],
        ["Woke up to find my car's tire flat. Just what I needed on a Monday morning! 🤦‍♂️ #MondayBlues"],
        ["Seriously regretting my decision to eat at that new restaurant last night. Food poisoning is not how I wanted to spend my weekend. 😷 #neveragain"],
        ["Just got a call from my boss saying I have to work overtime again this weekend. Can't catch a break! 😞 #worklife"],
        ["Feeling so lonely lately. It's hard to see all my friends posting about their happy lives while I'm stuck here feeling miserable. 😔 #lonely"]
                 ], dtype=str)

neutral_test = np.array([["Just finished my morning coffee. Ready to tackle the day ahead! ☕️ #morningroutine"],
        ["Enjoying a leisurely walk in the park. The weather is perfect today! 🌳☀️ #naturelovers"],
        ["Attending a webinar on the latest trends in technology. Excited to learn something new! 💻📚 #lifelonglearner"],
        ["Spent the afternoon organizing my bookshelf. Feels good to declutter and tidy up! 📚✨ #organization"],
        ["Cooked a simple dinner at home tonight. Sometimes the simplest meals are the most satisfying. 🍽️ #homecooking"]
                 ], dtype=str)


class Model:
    #   name negative_neutral_positive_prediction_1.1.0.h5
    def __init__(self, name):
        self.loaded_model = self.load_model(name)

    def load_model(self, name):
        path = rf"..\model\{name}.h5"
        loaded_model = keras.models.load_model(path)
        return loaded_model



class TextVectorizationModel(Model):

    vocab_path = r"..\model\vocab_1.2.1.txt"
    encoding = "utf-8"

    def __init__(self, name):
        super().__init__(name)
        self.text_vectorization_layer = self.loaded_model.layers[0]
        self.set_dictionary()

    def set_dictionary(self):
        file_contents = []
        with open(TextVectorizationModel.vocab_path, 'r', encoding=TextVectorizationModel.encoding) as f:
            for line in f:
                file_contents.append(line.strip())
        self.text_vectorization_layer.set_vocabulary(file_contents)

    def vectorize_input(self, tensor_input):
        vectorized_input = self.text_vectorization_layer(tensor_input)
        return vectorized_input



class SentimentPredictionModel(Model):

    def __init__(self, name, text_vectorization_model_name):
        super().__init__(name)
        self.text_vectorization_model = TextVectorizationModel(text_vectorization_model_name)

    def predict_sentiment(self, text_input: str):
        numpy_input = np.array([text_input], dtype=str)
        tensor_input = tf.convert_to_tensor(numpy_input, dtype=tf.string)
        vectorized_input = self.text_vectorization_model.vectorize_input(tensor_input)

        def _fixup_shape(text):
            text.set_shape([1, 25])
            return text
            
        vectorized_input = _fixup_shape(vectorized_input)

        score = self.loaded_model(vectorized_input)[0].numpy()
        max_score_index = np.argmax(score)
        max_score = np.max(score)

        if max_score_index == 0:
            return "NEGATIVE", max_score
        elif max_score_index == 1:
            return "NEUTRAL", max_score
        else:
            return "POSITIVE", max_score
        

    def predict_speech_sentiment(self, speech_input):
        prediction, probability = self.predict_sentiment(speech_input)
        return f"{prediction} sentiment in {probability*100:.3f} %"



class Chatbot:

    sentiment_prediction_model_name = "negative_neutral_positive_prediction_1.2.1"
    text_vectorization_model_name = "text_vectorization_model_1.2.1"

    def __init__(self):
        self.sentiment_prediction_model = SentimentPredictionModel(
                    Chatbot.sentiment_prediction_model_name,
                    Chatbot.text_vectorization_model_name
                    )
        

    def predict_sentiment(self, input):
        prediction, probability = self.sentiment_prediction_model.predict_sentiment(input)
        response = f"{prediction} sentiment in {probability*100:.3f} %"
        print(response)
        return response