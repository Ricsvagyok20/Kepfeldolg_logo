import cv2
import numpy as np
import os
from keypoint_detection.pureObject import pureObject
#from pureObject import pureObject

def preprocess_image(image, need_edges):
    if image is None:
        return None

    h, w = image.shape[:2]
    target_width, target_height = 400,400
    
    scale = min(target_width / w, target_height / h)
    
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = cv2.GaussianBlur(image, (5, 5), 0)

    if need_edges:
        image = cv2.Canny(image, 150, 250)
   
    return image

def detect_keypoints(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    
    strong_keypoints = [kp for kp in keypoints if kp.response > 0]
    strong_descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])

    sorted_keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

    strong_keypoints = sorted_keypoints[:30]

    strong_descriptors = np.array([descriptors[i] for i, kp in enumerate(keypoints) if kp in strong_keypoints])
    
    return strong_keypoints, strong_descriptors

def check_geometric_consistency(logo_keypoints, target_keypoints, matches, distance_threshold):

    consistent_matches = []
    
    for i, match1 in enumerate(matches):
        is_consistent = True
        logo_pt1 = np.array(logo_keypoints[match1.queryIdx].pt)
        target_pt1 = np.array(target_keypoints[match1.trainIdx].pt)
        
        for j, match2 in enumerate(matches):
            if i == j:
                continue

            logo_pt2 = np.array(logo_keypoints[match2.queryIdx].pt)
            target_pt2 = np.array(target_keypoints[match2.trainIdx].pt)
            
            logo_distance = np.linalg.norm(logo_pt1 - logo_pt2)
            target_distance = np.linalg.norm(target_pt1 - target_pt2)
            
            if abs(logo_distance - target_distance) > distance_threshold:
                is_consistent = False
                break
        
        if is_consistent:
            consistent_matches.append(match1)
    
    return consistent_matches

def match_against_pure(pure_descriptors, descriptors,pure_keypoints,keypoints):
    if pure_descriptors is None or descriptors is None or len(pure_descriptors) == 0 or len(descriptors) == 0:
        print("Descriptors are empty, no matching possible")
        return 0, [] 

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.match(pure_descriptors, descriptors)

    matches = sorted(matches, key=lambda x: x.distance)[:20]
    consistent_matches = check_geometric_consistency(pure_keypoints, keypoints, matches, 350)

    match_quality = np.mean([m.distance for m in consistent_matches]) if consistent_matches else 0
    
    return match_quality, consistent_matches


def get_best_matching_logo(pure1, pure2):
    best_logo = None
    highest_average = float('-inf')  
    
    for logo in pure1.logos.keys():
        match_result1 = pure1.logos[logo]["match_result"]
        match_result2 = pure2.logos[logo]["match_result"]
        
        if match_result1 is None or match_result2 is None:
            continue
        
        average_result = (match_result1 + match_result2) / 2
        
        if average_result > highest_average:
            highest_average = average_result
            best_logo = logo    

    return best_logo



def predict_with_keypoint(image = None):


    ##############################
    production = True
    ##############################

    image_directory = 'chamfer'
    output_directory = 'keypoint_detection\\output_images'


    #tökéletes kép
    pure_object = pureObject()
    pure_image = preprocess_image(cv2.imread('assets/logos/apple/apple_pure.jpg'), False)
    pure_object.logos["apple"]["keypoints"], pure_object.logos["apple"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["apple"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/logos/honda/honda_logo_main.jpg'), False)
    pure_object.logos["honda"]["keypoints"], pure_object.logos["honda"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["honda"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/logos/nike/nike_logo_pure.jpg'), False)
    pure_object.logos["nike"]["keypoints"], pure_object.logos["nike"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["nike"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/logos/peugeot/peugeot_logo_main.jpg'), False)
    pure_object.logos["peugeot"]["keypoints"], pure_object.logos["peugeot"]["descriptors"] = detect_keypoints(pure_image)
    pure_object.logos["peugeot"]["image"] = pure_image

    #chamfer image
    pure_object_chamfer = pureObject()
    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/apple_chamfer_template.png'), False)
    pure_object_chamfer.logos["apple"]["keypoints"], pure_object_chamfer.logos["apple"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["apple"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/honda_chamfer_template.png'), False)
    pure_object_chamfer.logos["honda"]["keypoints"], pure_object_chamfer.logos["honda"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["honda"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/nike_chamfer_template.png'), False)
    pure_object_chamfer.logos["nike"]["keypoints"], pure_object_chamfer.logos["nike"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["nike"]["image"] = pure_image

    pure_image = preprocess_image(cv2.imread('assets/chamfer_templates/peugeot_chamfer_template.png'), False)
    pure_object_chamfer.logos["peugeot"]["keypoints"], pure_object_chamfer.logos["peugeot"]["descriptors"] = detect_keypoints(pure_image)
    pure_object_chamfer.logos["peugeot"]["image"] = pure_image


    if(production):
        processed_image = preprocess_image(image,False)
        processed_image_chamfer = preprocess_image(image,True)
        if processed_image is not None:
            keypoints, descriptors = detect_keypoints(processed_image)
            keypoints_chamfer, descriptors_chamfer = detect_keypoints(processed_image_chamfer)
            for logo, data in pure_object.logos.items():
                data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors,data["keypoints"], keypoints)
           
            for logo, data in pure_object_chamfer.logos.items():
                data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors_chamfer,data["keypoints"], keypoints_chamfer)    


            matched_img = cv2.drawMatches(pure_object.logos[best_logo]["image"], pure_object.logos[best_logo]["keypoints"], processed_image, keypoints, pure_object.logos[best_logo]["matches"][:50], None, flags=2)
            return best_logo, matched_img
    else:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(image_directory):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            processed_image = preprocess_image(image, False)
            processed_image_chamfer = preprocess_image(image,True)
            if processed_image is not None:
                keypoints, descriptors = detect_keypoints(processed_image)
                keypoints_chamfer, descriptors_chamfer = detect_keypoints(processed_image_chamfer)
                for logo, data in pure_object.logos.items():
                    data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors,data["keypoints"], keypoints)


                for logo, data in pure_object_chamfer.logos.items():
                    data["match_result"], data["matches"] = match_against_pure( data["descriptors"], descriptors_chamfer,data["keypoints"], keypoints_chamfer)    

                best_logo = get_best_matching_logo(pure_object_chamfer, pure_object)


                matched_img = cv2.drawMatches(pure_object.logos[best_logo]["image"], pure_object.logos[best_logo]["keypoints"], processed_image, keypoints, pure_object.logos[best_logo]["matches"][:50], None, flags=2)
                output_path = os.path.join('keypoint_detection\\output_images', f"matched_keypoints_{filename}")
                cv2.imwrite(output_path, matched_img)


if __name__ == "__main__":
    predict_with_keypoint()

