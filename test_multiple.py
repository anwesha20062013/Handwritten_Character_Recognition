import numpy as np

from predictor import DigitPredictor

predictor = DigitPredictor()

imgs = []

for i in range(3):

    imgs.append(
        np.zeros((28,28),dtype=np.float32)
    )

number,confidence = predictor.predict_multiple(imgs)

print(number)

print(confidence)