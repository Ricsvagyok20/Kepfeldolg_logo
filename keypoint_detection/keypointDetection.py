import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Kontraszt javítás (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Zajszűrés (Gaussian blur)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image

def detect_keypoints(image):

    detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)

    strong_keypoints = [kp for kp in keypoints if kp.response > 0]
    descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])
    
    return strong_keypoints, descriptors

def match_against_pure(pure_image, pure_keypoints, pure_descriptors, image, keypoints, descriptors, filename):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(pure_descriptors, descriptors)

    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 300000000] 

    matched_img = cv2.drawMatches(pure_image, pure_keypoints, image, keypoints, good_matches[:10], None, flags=2)

    output_path = os.path.join('keypoint_detection\\output_images', f"matched_keypoints_{filename}")
    cv2.imwrite(output_path, matched_img)


def main():
    # Kép mappájának megadása
    image_directory = 'assets\\logos\\apple\\preproc'
    output_directory = 'keypoint_detection\\output_images'

    #tökéletes kép
    pure_image = preprocess_image('assets/logos/apple/apple_pure.jpg')
    pure_keypoints, pure_descriptors = detect_keypoints(pure_image)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Képek feldolgozása a mappában
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        processed_image = preprocess_image(image_path)

        if processed_image is not None:
            keypoints, descriptors = detect_keypoints(processed_image)
            match_against_pure(pure_image, pure_keypoints, pure_descriptors, processed_image, keypoints, descriptors, filename)


 
    

if __name__ == "__main__":
    main()
