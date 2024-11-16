import cv2

def load_and_preprocess_image(image_path, target_size=None):
    """
    Kép betöltése és opcionális átméretezése.
    
    :param image_path: A kép elérési útvonala.
    :param target_size: Ha megadva, a kép méretarányosan átméretezésre kerül (max oldalhossz).
    :return: Az előfeldolgozott kép.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"A kép nem található: {image_path}")

    if target_size:
        image = resize_with_aspect_ratio(image, target_size)

    return image

def resize_with_aspect_ratio(image, target_size):
    """
    Méretarányos átméretezés a kép hosszabbik oldala alapján.

    :param image: Az átméretezendő kép.
    :param target_size: A célhossz a hosszabbik oldalra.
    :return: Az átméretezett kép.
    """
    h, w = image.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image
