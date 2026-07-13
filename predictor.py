import tensorflow as tf
import numpy as np

from preprocess import preprocess_digit
from character_mapping import load_mapping


class Predictor:

    def __init__(self, mode="digit"):

        self.mode = mode

        if mode == "digit":

            self.model = tf.keras.models.load_model(
                "models/handwritten_cnn_model.h5",
                compile=False
            )

        else:

            self.model = tf.keras.models.load_model(
                "models/character_model.h5",
                compile=False
            )

            self.mapping = load_mapping()

    def predict(self, digit_images):

        predictions = []
        confidences = []

        for digit in digit_images:

            img = preprocess_digit(digit)

            pred = self.model.predict(img, verbose=0)

            idx = int(np.argmax(pred))

            confidence = float(np.max(pred) * 100)

            if self.mode == "digit":

                result = str(idx)

            else:

                result = self.mapping[idx]

            predictions.append(result)

            confidences.append(confidence)

        return predictions, confidences