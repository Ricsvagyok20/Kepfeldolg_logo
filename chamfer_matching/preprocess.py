from tempfile import template

import cv2 as cv
import numpy as np

from chamfer_matching.template import chamfer_template
from chamfer_matching.template import generate_scaled_templates


def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv.imread(path)
        images.append(image)
    return images

def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]

    # Determine the scaling factor based on the longest side
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized_image


def preprocess_chamfer(image):
    # Szürkeárnyalatos konverzió
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Éldetektálás
    edges = cv.Canny(gray_image, 250, 600)

    # Binarizálás
    _, binary_image = cv.threshold(edges, 127, 255, cv.THRESH_BINARY)
    return binary_image


def chamfer_match(template, learning_image):
    # Compute the distance transform on the learning image
    distance_map = cv.distanceTransform(learning_image, cv.DIST_L2, 3)

    # Visualize the Chamfer matching score map
    # cv.imshow("Template", template)
    # print("Visualizing Matching Score Map...")
    # matching_score = visualize_matching_score(distance_map, learning_image)

    # Perform Chamfer matching by sliding the template
    matching_score = cv.filter2D(distance_map, -1, template.astype(np.float32))

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matching_score)
    return min_val, min_loc


def process_images(image_paths, template_path):
    images = load_images(image_paths)
    processed_images = []

    binary_template_original = chamfer_template(template_path)
    binary_template = resize_with_aspect_ratio(binary_template_original, 256)

    for image in images:
        # Skálázás képarányok megtartásával, hogy a logó eredeti formájában maradjon
        image = resize_with_aspect_ratio(image, 512)
        binary_image = preprocess_chamfer(image)

        cv.imshow("Chamfer template", binary_template)

        cv.imshow("New image binary", binary_image)

        best_score = float('inf')
        best_location = None
        best_template_size = None

        # Iterate over scaled templates
        for scaled_template in generate_scaled_templates(binary_template):
            score, location = chamfer_match(scaled_template, binary_image)
            if score < best_score:
                best_score = score
                best_location = location
                best_template_size = scaled_template.shape

        # Draw a circle or bounding box around the best match on the original image
        top_left = best_location
        h, w = best_template_size
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw a rectangle or a circle around the detected region
        cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Optionally, crop the matched region
        matched_region = image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        # Display the result
        cv.imshow("Best Match", image)
        cv.imshow("Matched Region", matched_region)
        cv.waitKey(0)
        cv.destroyAllWindows()

        processed_images.append(matched_region)

    return np.array(processed_images)