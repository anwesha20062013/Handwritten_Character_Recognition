import cv2
import numpy as np


def preprocess_digit(img):

    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary image
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    # Find bounding box
    coords = cv2.findNonZero(img)

    if coords is None:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    digit = img[y:y+h, x:x+w]

    # Keep aspect ratio
    size = 20

    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    digit = cv2.resize(digit, (new_w, new_h))

    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    # Normalize
    canvas = canvas.astype("float32") / 255.0

    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas