import cv2
import numpy as np


def segment_digits(image):

    # Convert RGBA/BGR to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # White digit -> white pixels
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Projection profile
    vertical_sum = np.sum(thresh, axis=0)

    digit_regions = []

    inside = False
    start = 0

    for i, value in enumerate(vertical_sum):

        if value > 0 and not inside:
            start = i
            inside = True

        elif value == 0 and inside:
            end = i

            if end - start > 5:
                digit_regions.append((start, end))

            inside = False

    if inside:
        digit_regions.append((start, len(vertical_sum)-1))

    digits = []

    for start, end in digit_regions:

        digit = thresh[:, start:end]

        ys, xs = np.where(digit > 0)

        if len(xs) == 0:
            continue

        digit = digit[
            ys.min():ys.max()+1,
            xs.min():xs.max()+1
        ]

        h, w = digit.shape

        size = max(h, w)

        square = np.zeros((size, size), dtype=np.uint8)

        yoff = (size-h)//2
        xoff = (size-w)//2

        square[yoff:yoff+h, xoff:xoff+w] = digit

        square = cv2.resize(square, (28, 28))

        digits.append(square)

    return digits