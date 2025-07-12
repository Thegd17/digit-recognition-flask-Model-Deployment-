import cv2
import numpy as np

def preprocess_image(file_stream):
    img = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  # MNIST digits are white on black
    img = img.astype("float32") / 255
    img = img.reshape(1, 28, 28, 1)
    return img
