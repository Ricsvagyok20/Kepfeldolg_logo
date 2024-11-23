import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pureObject import pureObject

def preprocess_image(image):
    if image is None:
        return None

    h, w = image.shape[:2]
    target_width, target_height = 400,400
    
    # Compute the scaling factor
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # Resize the image
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Kontraszt javítás (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Zajszűrés (Gaussian blur)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image

"""def detect_keypoints(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)

    strong_keypoints = [kp for kp in keypoints if kp.response > 0]
    descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])
    
    return strong_keypoints, descriptors

def match_against_pure(pure_descriptors, descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(pure_descriptors, descriptors, k=2)

   
    match_quality = sum(1.0 / m.distance for m in good_matches) if good_matches else 0
    return match_quality, good_matches"""

def detect_keypoints(image, response_threshold=0.002):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    
    print(f"Detected {len(keypoints)} keypoints")  # Debugging line
    
    # Filter keypoints based on their response
    strong_keypoints = [kp for kp in keypoints if kp.response > response_threshold]
    strong_descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])

    print(f"After filtering, {len(strong_keypoints)} keypoints remain")  # Debugging line
    
    return strong_keypoints, strong_descriptors


def match_against_pure(pure_descriptors, descriptors):
    if pure_descriptors is None or descriptors is None or len(pure_descriptors) == 0 or len(descriptors) == 0:
        print("Descriptors are empty, no matching possible")
        return 0, []  # Return no matches if descriptors are invalid

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Perform knn matching
    matches = bf.match(pure_descriptors, descriptors)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate match quality as the average distance
    match_quality = np.mean([m.distance for m in matches]) if matches else 0

    return match_quality, matches




def predict_with_keypoint(image):

    output_directory = 'keypoint_detection\\output_images'

    #tökéletes kép
    pure_object = pureObject()
    pure_image = preprocess_image('assets/logos/apple/apple_pure.jpg')
    pure_object.logos["apple"]["keypoints"], pure_object.logos["apple"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["apple"]["image"] = pure_image

    pure_image = preprocess_image('assets/logos/honda/honda_logo_main_for_chamfer.jpg')
    pure_object.logos["honda"]["keypoints"], pure_object.logos["honda"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["honda"]["image"] = pure_image

    pure_image = preprocess_image('assets/logos/nike/nike_logo_pure.jpg')
    pure_object.logos["nike"]["keypoints"], pure_object.logos["nike"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["nike"]["image"] = pure_image

    pure_image = preprocess_image('assets/logos/peugeot/peugeot_logo_23.jpg')
    pure_object.logos["peugeot"]["keypoints"], pure_object.logos["peugeot"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["peugeot"]["image"] = pure_image

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    processed_image = preprocess_image(image)

    if processed_image is not None:
         keypoints, descriptors = detect_keypoints(processed_image)
         for logo, data in pure_object.logos.items():
            data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors)

         max_match_result = -float('inf')  
         best_logo = None  
         for logo, data in pure_object.logos.items():
            if data["match_result"] > max_match_result:
                max_match_result = data["match_result"]
                best_logo = logo
         
         matched_img = cv2.drawMatches(pure_object.logos[best_logo]["image"], pure_object.logos[best_logo]["keypoints"], processed_image, keypoints, pure_object.logos[best_logo]["matches"][:50], None, flags=2)
         return best_logo, matched_img



"""""
if __name__ == "__main__":
    predict_with_keypoint()
"""
