import cv2 as cv

def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]

    # Determine the scaling factor based on the longest side
    if h > w:
        new_h = int(target_size)
        new_w = int(w * (target_size / h))
    else:
        new_w = int(target_size)
        new_h = int(h * (target_size / w))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized_image