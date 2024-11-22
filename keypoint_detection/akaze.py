import cv2
import numpy as np
from preprocess import load_and_preprocess_image

# Képek előfeldolgozása
logo_image = load_and_preprocess_image('./../assets/logos/nike/nike_logo_pure.jpg', target_size=300)
target_image = load_and_preprocess_image('./../assets/logos/nike/nike_logo_21.jpg', target_size=800)

# AKAZE objektum létrehozása
akaze = cv2.AKAZE_create()

# Kulcspontok és leírók kinyerése
keypoints_logo, descriptors_logo = akaze.detectAndCompute(logo_image, None)
keypoints_target, descriptors_target = akaze.detectAndCompute(target_image, None)

# Kulcspontok összehasonlítása Brute-Force matcherrel
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_logo, descriptors_target)

# Találatok rendezése távolság alapján
matches = sorted(matches, key=lambda x: x.distance)

# Ha elegendő találat van, homográfiát számolunk
if len(matches) > 10:
    src_pts = np.float32([keypoints_logo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Határoló doboz kirajzolása
    # h, w = logo_image.shape
    # points = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    # transformed_points = cv2.perspectiveTransform(points, matrix)

    # Téglalap zöld színnel (BGR: (0, 255, 0))
    # target_image_with_box = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
    # cv2.polylines(target_image_with_box, [np.int32(transformed_points)], True, (0, 255, 0), 3)
    # cv2.imshow('Detected Logo', target_image_with_box)
else:
    print("Nincs elegendő egyezés!")

# Eredmény megjelenítése
result_image = cv2.drawMatches(logo_image, keypoints_logo, target_image, keypoints_target, matches[:20], None, flags=2)
cv2.imshow('Matches', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
