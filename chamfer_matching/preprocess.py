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
    edges = cv.Canny(gray_image, 250, 400) # Lehet itt is akkor valami 200-300hoz hasonló mehetne
    # 250-600-al elég jó a 200-300 néhol segít, de a túl sok él is gondot okoz
    # 200-400 nem rossz, 200-300 se rossz

    # Binarizálás
    _, binary_image = cv.threshold(edges, 0, 255, cv.THRESH_BINARY)
    return binary_image


def chamfer_match(template, learning_image):
    dist_transform = cv.distanceTransform(cv.bitwise_not(learning_image), cv.DIST_L2, 3)
    # dist_transform = dist_transform.astype(np.float32)

    # Sliding window Chamfer matching
    h, w = template.shape
    min_distance = float('inf')
    best_location = (0, 0)

    for y in range(learning_image.shape[0] - h + 1):
        for x in range(learning_image.shape[1] - w + 1):
            # Extract the window from the distance transform that matches the template's size
            roi = dist_transform[y:y + h, x:x + w]

            # Mask out non-edges in the template for distance summing
            dist_sum = np.sum(roi[template > 0])

            # Check if we found a closer match
            if dist_sum < min_distance:
                min_distance = dist_sum
                best_location = (x, y)

    # print("Best match location:", best_location)
    # print("Minimum Chamfer distance:", min_distance)

    #Normalizálom a scoret, segített
    min_distance /= h * w

    if best_location[0] >= 10 and best_location[1] >= 10:
        x = best_location[0] - 10
        y = best_location[1] - 10
        best_location = (x, y)

    return min_distance, best_location

def best_match_image(image, binary_image, binary_template):
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
    if best_template_size is None:
        return None, None

    top_left = best_location
    h, w = best_template_size
    if best_template_size[0] != 512 and best_template_size[1] != 512:
        h += 20
        w += 20
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle or a circle around the detected region
    cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    matched_region = image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
    return image, matched_region

def process_images(image_paths, template_path):
    images = load_images(image_paths)
    processed_images = []

    binary_template = chamfer_template(template_path)

    for image in images:

        # Skálázás képarányok megtartásával, hogy a logó eredeti formájában maradjon
        image = resize_with_aspect_ratio(image, 512)
        binary_image = preprocess_chamfer(image)

        # cv.imshow("Chamfer template", binary_template)
        # cv.imshow("New image binary", binary_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        image, matched_region = best_match_image(image, binary_image, binary_template)

        if image is None:
            continue

        # Display the result
        # cv.imshow("Best Match", image)
        # cv.imshow("Matched Region", matched_region)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        processed_images.append(matched_region)

    target_shape = (512, 512, 3)  # Assuming 3 channels (RGB)

    # Pad all images to the target shape
    padded_images = [pad_image(img, target_shape) for img in processed_images]

    return_images = np.array(padded_images)

    return_images = return_images / 255.0

    return return_images