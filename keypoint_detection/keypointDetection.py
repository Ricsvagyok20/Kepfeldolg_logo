import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    imgray = cv2.dilate(imgray, struct)
    imgray = cv2.erode(imgray, struct)
    imgray = cv2.erode(imgray, struct)
    imgray = cv2.dilate(imgray, struct)

    # Kontraszt javítás (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(imgray)

    # Zajszűrés (Gaussian blur)
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    return blurred_image

def detect_keypoints():

    image = preprocess_image('keypoint_detection/assets/logos/apple/preproc/apple_41.jpg')

    detector = cv2.ORB_create()

    # Kulcspontok és jellemzők kinyerése
    keypoints, descriptors = detector.detectAndCompute(image, None)
    
    image2 = preprocess_image('keypoint_detection/assets/logos/apple/preproc/apple_39.jpg')

    if image is None:
        print("Error: 'logo1.jpg' not found or could not be loaded.")
        return
    if image2 is None:
        print("Error: 'logo2.jpg' not found or could not be loaded.")
        return

    orb = cv2.ORB_create()
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(image, keypoints, image2, keypoints2, matches[:10], None, flags=2)
    plt.imshow(matched_img)
    plt.axis('off')  # Az axis eltávolítása a tisztább megjelenítés érdekében
    plt.show()

    # Kép kirajzolása a kulcspontokkal (vizualizációhoz)
    #output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #return output_image, keypoints, descriptors

def main():
    # Kép mappájának megadása
    image_directory = 'assets\\logos\\apple\\preproc'
    output_directory = 'keypoint_detection\\output_images'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Képek feldolgozása a mappában
    """"for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        processed_image = preprocess_image(image_path)

        if processed_image is not None:
            output_image, keypoints, descriptors = detect_keypoints(processed_image)
            print(f"Detected {len(keypoints)} keypoints in {filename} using ORB.")

            # Eredmény elmentése
            output_path = os.path.join(output_directory, f"keypoints_{filename}")
            cv2.imwrite(output_path, output_image)
            print(f"Processed image saved to {output_path}")
"""
    detect_keypoints()
    

if __name__ == "__main__":
    main()
