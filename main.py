import cv2

import config

from chamfer_matching.preprocess import process_images

# Fő program
processed_images = []

# Iterate through each logo path and its corresponding template
for logo_group in config.LOGO_PATHS_WITH_TEMPLATES:
    logo_paths = logo_group['logo_paths']
    template_path = logo_group['template_path']

    # Process images for the current logo group
    group_processed_images = process_images(logo_paths, template_path)
    processed_images.extend(group_processed_images)

# img = cv2.imread('assets/logos/nike/nike_logo_pure.jpg')
# cv2.imshow("asd", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(len(processed_images))