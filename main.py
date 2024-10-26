print("asd")
import config

from chamfer_matching.preprocess import process_images

# Fő program
image_paths = [path for sublist in config.LOGO_PATHS for path in sublist]  # Egyesítjük az összes útvonalat egy listába
processed_images = process_images(image_paths)

# Normalizálás
processed_images = processed_images / 255.0