import cv2
import numpy as np
import os

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

def detect_keypoints(image):
    detector = cv2.ORB_create()

    # Kulcspontok és jellemzők kinyerése
    keypoints, descriptors = detector.detectAndCompute(image, None)

    # Kép kirajzolása a kulcspontokkal (vizualizációhoz)
    output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return output_image, keypoints, descriptors

def main():
    # Kép mappájának megadása
    image_directory = 'assets\\logos\\apple\\preproc'
    output_directory = 'keypoint_detection\\output_images'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Képek feldolgozása a mappában
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        processed_image = preprocess_image(image_path)

        if processed_image is not None:
            output_image, keypoints, descriptors = detect_keypoints(processed_image)
            print(f"Detected {len(keypoints)} keypoints in {filename} using ORB.")

            # Eredmény elmentése
            output_path = os.path.join(output_directory, f"keypoints_{filename}")
            cv2.imwrite(output_path, output_image)
            print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    main()
