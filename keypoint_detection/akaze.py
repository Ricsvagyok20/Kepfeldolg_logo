import cv2
import numpy as np
from preprocess import load_and_preprocess_image 

def draw_bounding_box_around_keypoints(logo_image, target_image, keypoints_logo, keypoints_target, matches):
    """
    Rajzoljon egy téglalapot a célképen, amely a legnagyobb kulcspont-sűrűségű terület körül van.
    """
    # Kulcspontok pozíciójának kiemelése
    pts = np.float32([keypoints_target[m.trainIdx].pt for m in matches])
    
    # A középpont meghatározása
    center = np.mean(pts, axis=0)
    
    # A kulcspontok terjedelme
    max_distance = np.max([np.linalg.norm(center - pt) for pt in pts])
    
    # A téglalap szélei a középpont körül
    margin = max_distance * 0.5
    x_min = int(center[0] - margin)
    y_min = int(center[1] - margin)
    x_max = int(center[0] + margin)
    y_max = int(center[1] + margin)
    
    # Téglalap rajzolása a célképen
    target_image_with_box = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(target_image_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    return target_image_with_box

# Képek betöltése és előfeldolgozása
logo_image = load_and_preprocess_image('./../assets/logos/nike/nike_logo_pure.jpg', target_size=300)
target_image = load_and_preprocess_image('./../assets/logos/nike/nike_logo_12.jpg', target_size=800)

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
if len(matches) > 4:
    src_pts = np.float32([keypoints_logo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Homográfia számítása
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Rajzolás a középpont körüli téglalap körül
    target_image_with_box = draw_bounding_box_around_keypoints(logo_image, target_image, keypoints_logo, keypoints_target, matches)
    cv2.imshow('Detected Logo with Bounding Box', target_image_with_box)

else:
    print("Nincs elegendő egyezés!")

# Egyezések vizualizációja
result_image = cv2.drawMatches(logo_image, keypoints_logo, target_image, keypoints_target, matches[:20], None, flags=2)
cv2.imshow('Matches', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
