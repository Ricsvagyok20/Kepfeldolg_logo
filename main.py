import config

from chamfer_matching.preprocess import process_images

# Fő program
image_paths = [path for sublist in config.LOGO_PATHS for path in sublist]  # Egyesítjük az összes útvonalat egy listába
template_path = 'assets/logos/honda/honda_logo_main_for_chamfer.jpg'
processed_images = process_images(image_paths, template_path)

# Normalizálás
processed_images = processed_images / 255.0