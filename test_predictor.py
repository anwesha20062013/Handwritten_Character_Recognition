import numpy as np
from predictor import DigitPredictor

predictor = DigitPredictor()

dummy = np.zeros((28,28),dtype=np.float32)

digit,confidence,_ = predictor.predict(dummy)

print(digit)
print(confidence)