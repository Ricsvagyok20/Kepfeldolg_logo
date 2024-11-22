import glob

# Logókhoz vezető útvonalak
LOGO_PATHS_WITH_TEMPLATES = [
    {
        'logo_paths': glob.glob('assets/logos/honda/*.jpg'),
        'template_path': 'assets/logos/honda/honda_logo_main_for_chamfer.jpg'
    },
    # {
    #     'logo_paths': glob.glob('assets/logos/apple/*.jpg'),
    #     'template_path': 'assets/logos/apple/apple_pure.jpg'
    # },
    # {
    #     'logo_paths': glob.glob('assets/logos/nike/*.jpg'),
    #     'template_path': 'assets/logos/nike/nike_logo_pure.jpg'
    # },
    # {
    #     'logo_paths': glob.glob('assets/logos/peugeot/*.jpg'),
    #     'template_path': 'assets/logos/peugeot/peugeot_logo_23.jpg'
    # },
]
