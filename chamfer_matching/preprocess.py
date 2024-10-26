import cv2
import numpy as np


def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        images.append(image)
    return images


def preprocess_chamfer(image, size=(256, 256)):
    # Skálázás
    # scaled_image = cv2.resize(image, size)

    # Szürkeárnyalatos konverzió
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Éldetektálás
    edges = cv2.Canny(gray_image, 100, 200)

    # Binarizálás
    _, binary_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    return binary_image


# Chamfer-illesztés
# cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
# TODO: a legjobban match-elő chamfer terület kivágása


def process_images(image_paths):
    images = load_images(image_paths)
    processed_images = []

    for image in images:
        binary_image = preprocess_chamfer(image)
        # ide jön a chamfer-illesztés hívás
        # chamfer_image = method call
        # processed_images.append(chamfer_image)

    return np.array(processed_images)