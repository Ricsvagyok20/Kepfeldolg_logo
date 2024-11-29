import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pureObject import pureObject

def preprocess_image(image):
    if image is None:
        return None
    
    h, w = image.shape[:2]
    target_width, target_height = 400, 400
    scale = min(target_width / w, target_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def detect_keypoints(image, response_threshold=0.002, max_keypoints=20):
    akaze = cv2.AKAZE_create(threshold=0.01)
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    
    if not keypoints or descriptors is None:
        print("No keypoints detected")
        return [], None

    print(f"Detected {len(keypoints)} keypoints")

    sorted_keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

    strong_keypoints = sorted_keypoints[:max_keypoints]

    strong_descriptors = descriptors[:len(strong_keypoints)]

    print(f"After filtering, {len(strong_keypoints)} keypoints remain") 

    for i, kp in enumerate(strong_keypoints):
        print(f"Keypoint {i}: Response = {kp.response}")

    return strong_keypoints, strong_descriptors



def match_against_pure(pure_descriptors, descriptors):
    if pure_descriptors is None or descriptors is None or len(pure_descriptors) == 0 or len(descriptors) == 0:
        print("Descriptors are empty, no matching possible")
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.knnMatch(pure_descriptors, descriptors, k=1)

    if not matches:
        print("No matches found")
        return 0, []

    matches = sorted(matches, key=lambda x: x[0].distance)  

    match_quality = np.mean([m[0].distance for m in matches]) if matches else 0

    return match_quality, matches




def get_best_matching_logo(pure1, pure2):
    best_logo = None
    highest_average = float('-inf')  
    
    for logo in pure1.logos.keys():
        # Extract match results from both instances
        match_result1 = pure1.logos[logo]["match_result"]
        match_result2 = pure2.logos[logo]["match_result"]
        
        # Ensure values exist (handle None gracefully)
        if match_result1 is None or match_result2 is None:
            continue
        
        # Calculate the average match result
        average_result = (match_result1 + match_result2) / 2
        
        # Update the best logo if the current average is higher
        if average_result > highest_average:
            highest_average = average_result
            best_logo = logo    
    
    # Return the logo name with the highest average match result
    return best_logo



def predict_with_keypoint(image = None):
    ##############################
    production = False
    ##############################

    image_directory = 'output_images'
    output_directory = 'keypoint_detection\\output_images'


# pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/_chamfer_template.jpg'))
    #tökéletes kép
    pure_object = pureObject()
    pure_image = preprocess_image(cv2.imread('assets/logos/apple/apple_pure.jpg'))
    pure_object.logos["apple"]["keypoints"], pure_object.logos["apple"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["apple"]["image"] = pure_image

    #pure_image = preprocess_image(cv2.imread('assets/logos/honda/honda_logo_main_for_chamfer.jpg'))
    pure_image = preprocess_image(cv2.imread('assets/logos/apple/apple_pure.jpg'))
    pure_object.logos["honda"]["keypoints"], pure_object.logos["honda"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["honda"]["image"] = pure_image

    #pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/nike_chamfer_template.png'))
    pure_image = preprocess_image(cv2.imread('assets/logos/nike/nike_logo_pure.jpg'))
    pure_object.logos["nike"]["keypoints"], pure_object.logos["nike"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["nike"]["image"] = pure_image

    #pure_image = preprocess_image(cv2.imread('assets/logos/peugeot/peugeot_logo_23.jpg'))
    pure_image = preprocess_image(cv2.imread('assets/logos/nike/nike_logo_pure.jpg'))
    pure_object.logos["peugeot"]["keypoints"], pure_object.logos["peugeot"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["peugeot"]["image"] = pure_image


    pure_object_chamfer = pureObject()
    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/apple_chamfer_template.png'))
    pure_object_chamfer.logos["apple"]["keypoints"], pure_object_chamfer.logos["apple"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["apple"]["image"] = pure_image

    #pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/honda_chamfer_template.png'))
    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/apple_chamfer_template.png'))
    pure_object_chamfer.logos["honda"]["keypoints"], pure_object_chamfer.logos["honda"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["honda"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/nike_chamfer_template.png'))
    pure_object_chamfer.logos["nike"]["keypoints"], pure_object_chamfer.logos["nike"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["nike"]["image"] = pure_image

    #pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/peugeot_chamfer_template.png'))
    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/nike_chamfer_template.png'))
    pure_object_chamfer.logos["peugeot"]["keypoints"], pure_object_chamfer.logos["peugeot"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["peugeot"]["image"] = pure_image


    if(production):
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
    else:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(image_directory):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            processed_image = preprocess_image(image)
            if processed_image is not None:
                keypoints, descriptors = detect_keypoints(processed_image)
                for logo, data in pure_object.logos.items():
                    data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors)


                for logo, data in pure_object_chamfer.logos.items():
                    data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors)    

                """
                max_match_result = -float('inf')  
                best_logo = None  
                for logo, data in pure_object.logos.items():
                    if data["match_result"] > max_match_result:
                        max_match_result = data["match_result"]
                        best_logo = logo
                 """

                best_logo = get_best_matching_logo(pure_object_chamfer, pure_object)


                matched_img = cv2.drawMatches(pure_object.logos[best_logo]["image"], pure_object.logos[best_logo]["keypoints"], processed_image, keypoints, pure_object.logos[best_logo]["matches"][:50], None, flags=2)
                output_path = os.path.join('keypoint_detection\\output_images', f"matched_keypoints_{filename}")
                cv2.imwrite(output_path, matched_img)

    




if __name__ == "__main__":
    predict_with_keypoint()