import numpy as np
import cv2

from chamfer_matching.cnn_model import predict_image


def visualiseData(classifier, testData, responses, title):
    # Create an image canvas for visualization
    width, height = 512, 512
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors for each class
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Add colors as needed
    # colors = [(255, 0, 0), (0, 255, 0)]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sampleMat = np.array([[j / width, i / height]], dtype=np.float32)
            # Predict using the CNN
            # prediction, predicted_label = predict_image(classifier, sampleMat)
            prediction = classifier.predict(sampleMat)  # Reshape for CNN
            if len(prediction) > 1:
                prediction = prediction[1]
            if prediction == 1:
                image[i, j] = colors[0]
            elif prediction == 2:
                image[i, j] = colors[1]
            elif prediction == 3:
                image[i, j] = colors[2]
            elif prediction == 4:
                image[i, j] = colors[3]

            # Color the pixel based on the predicted class
            # image[i, j] = colors[predicted_label]

    # Overlay test data points
    for idx, coords in enumerate(testData):
        x, y = int(coords[0] * width), int(coords[1] * height)
        true_label = np.argmax(responses[idx])  # Convert one-hot to class index
        cv2.circle(image, (x, y), 3, colors[true_label], -1)

    # Display the visualization
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()