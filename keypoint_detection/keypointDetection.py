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


    #orb
    #detector = cv2.ORB_create()

    #sift
    #sift = cv2.SIFT_create()

    #akaze
    akaze = cv2.AKAZE_create()


    #orb
    #keypoints, descriptors = detector.detectAndCompute(image, None)

    #sift
    #keypoints, descriptors = sift.detectAndCompute(image, None)

    #akaze
    keypoints, descriptors = akaze.detectAndCompute(image, None)

    strong_keypoints = [kp for kp in keypoints if kp.response > 0]
    descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])
    
    return strong_keypoints, descriptors

def match_against_pure(pure_image, pure_keypoints, pure_descriptors, image, keypoints, descriptors, filename):
    #orb
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #sift
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    #akaze
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    #orb, sift
    #matches = bf.match(pure_descriptors, descriptors)

    #akaze
    matches = bf.knnMatch(pure_descriptors, descriptors, k=2)

    #matches = sorted(matches, key=lambda x: x.distance)

    #orb, sift
    #good_matches = [m for m in matches if m.distance < 300]

    #akaze
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_quality = sum(1.0 / m.distance for m in good_matches) if good_matches else 0
    return match_quality, good_matches



def main():
    # Kép mappájának megadása
    image_directory = 'assets\\logos\\apple\\preproc'
    output_directory = 'keypoint_detection\\output_images'

    #tökéletes kép
    pure_image = preprocess_image('assets/logos/apple/apple_pure.jpg')
    pure_keypoints, pure_descriptors = detect_keypoints(pure_image)

    pure_image_inv = cv2.bitwise_not(pure_image)
    pure_keypoints_inv, pure_descriptors_inv = detect_keypoints(pure_image)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Képek feldolgozása a mappában
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        processed_image = preprocess_image(image_path)

        if processed_image is not None:
            keypoints, descriptors = detect_keypoints(processed_image)
            pure, good_matches = match_against_pure(pure_image, pure_keypoints, pure_descriptors, processed_image, keypoints, descriptors, filename)
            filename = 'inv' + filename 
            inv, good_matches = match_against_pure(pure_image_inv, pure_keypoints_inv, pure_descriptors_inv, processed_image, keypoints, descriptors, filename)

            if(pure > inv):
                matched_img = cv2.drawMatches(pure_image, pure_keypoints, processed_image, keypoints, good_matches[:50], None, flags=2)
                output_path = os.path.join('keypoint_detection\\output_images', f"matched_keypoints_{filename}")
                cv2.imwrite(output_path, matched_img)

            if(pure <= inv):
                matched_img = cv2.drawMatches(pure_image_inv, pure_keypoints_inv, processed_image, keypoints, good_matches[:50], None, flags=2)
                output_path = os.path.join('keypoint_detection\\output_images', f"matched_keypoints_{filename}")
                cv2.imwrite(output_path, matched_img)

            
            


 
    

if __name__ == "__main__":
    main()
