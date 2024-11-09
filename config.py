# config.py
import glob

# Logókhoz vezető útvonalak
LOGO_PATHS = [
    glob.glob('assets/logos/honda/*.jpg'),
    glob.glob('assets/logos/apple/preproc/*.jpg'),
    glob.glob('assets/logos/nike/*.jpg'),
    glob.glob('assets/logos/peugeot/*.jpg'),
    glob.glob('assets/logos/logitech/*.jpg')
]
