import cv2 as cv
import numpy as np

from chamfer_matching.template import chamfer_template
from chamfer_matching.template import generate_scaled_templates
from chamfer_matching.resize import resize_with_aspect_ratio


def pad_image(image, target_shape):
    pad_height = target_shape[0] - image.shape[0]
    pad_width = target_shape[1] - image.shape[1]
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')


def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv.imread(path)
        images.append(image)
    return images


def preprocess_chamfer(image):
    # Szürkeárnyalatos konverzió
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Éldetektálás
    edges = cv.Canny(gray_image, 200, 500) # Lehet itt is akkor valami 200-300hoz hasonló mehetne
    # 250-600-al elég jó a 200-300 néhol segít, de a túl sok él is gondot okoz
    # 200-400 nem rossz, 200-300 se rossz

    # Binarizálás
    _, binary_image = cv.threshold(edges, 0, 255, cv.THRESH_BINARY)
    return binary_image


def chamfer_match(template, learning_image):
    dist_transform = cv.distanceTransform(cv.bitwise_not(learning_image), cv.DIST_L2, 3)

    # Sliding window Chamfer matching
    h, w = template.shape
    min_distance, best_location = float('inf'), (0, 0)

    for y in range(learning_image.shape[0] - h + 1):
        for x in range(learning_image.shape[1] - w + 1):
            # Extract the window from the distance transform that matches the template's size
            roi = dist_transform[y:y + h, x:x + w]

            # Mask out non-edges in the template for distance summing
            dist_sum = np.sum(roi[template > 0])

            # Check if we found a closer match
            if dist_sum < min_distance:
                min_distance, best_location = dist_sum, (x, y)

    # print("Best match location:", best_location)
    # print("Minimum Chamfer distance:", min_distance)

    # Normalizálom a scoret, segített
    return min_distance / (h * w), best_location


def process_images(image_paths, template_path):
    images = load_images(image_paths)
    binary_template = chamfer_template(template_path)
    processed_images = []

    for image in images:
        # Skálázás képarányok megtartásával, hogy a logó eredeti formájában maradjon
        image = resize_with_aspect_ratio(image, 512)
        binary_image = preprocess_chamfer(image)

        # cv.imshow("Chamfer template", binary_template)
        # cv.imshow("New image binary", binary_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

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

        top_left = best_location
        h, w = best_template_size
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        matched_region = image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        # Display the result
        cv.imshow("Best Match", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        processed_images.append(matched_region)

    # Determine the target shape based on the largest image in processed_images
    target_shape = (max(img.shape[0] for img in processed_images), max(img.shape[1] for img in processed_images), 3)

    # Pad and normalize all images to the target shape
    return np.array([pad_image(img, target_shape) for img in processed_images]) / 255.0