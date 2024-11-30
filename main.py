import glob

import config
import numpy as np
import tkinter as tk
import random
import cv2 as cv

from chamfer_matching.preprocess import process_images
from chamfer_matching.resize import resize_with_aspect_ratio
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from chamfer_matching.cnn_model import build_model, train_model, predict_image
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keypoint_detection.keypointDetection import predict_with_keypoint

# Fő program
image_paths = []
original_images = []
processed_images = []
images = []

# Iterate through each logo path and its corresponding template
for i in glob.glob('processed_images/*.jpg'):
    image_paths.append(i)
    image = cv.imread(i)

    original_images.append(image)

normalized_images = np.array(original_images)

normalized_images = normalized_images / 255.0

processed_images = np.concatenate(normalized_images, axis=0)

counter = 0

# for logo_group in config.LOGO_PATHS_WITH_TEMPLATES:
#     logo_paths = logo_group['logo_paths']
#     image_paths.extend(logo_paths)
#     template_path = logo_group['template_path']
#
#     # Process images for the current logo group
#     group_processed_images, images = process_images(logo_paths, template_path)
#     for i in range(len(images)):
#         filename = 'processed_images/processed_image_' + str(counter) + '.jpg'
#         cv.imwrite(filename, images[i])
#         counter += 1

# Címkék generálása
labels = []
for i, logo_group in enumerate(config.LOGO_PATHS_WITH_TEMPLATES):
    logo_paths = logo_group['logo_paths']
    labels.extend([i] * len(logo_paths))

labels = np.array(labels)

label_dict = { 0: 'Honda', 1: 'Apple', 2: 'Nike', 3: 'Peugeot' }

# Keypoint detection accuracy mérés
# labels_keypoint = []
# for image in original_images:
#     prediction, image_2 = predict_with_keypoint(image)
#     labels_keypoint.append(prediction)
#
# keypoint_accuracy = 0
# print(len(labels), len(labels_keypoint))
# for i, label in enumerate(labels):
#     if label_dict[label].lower() == labels_keypoint[i]:
#         keypoint_accuracy += 1
# keypoint_accuracy = keypoint_accuracy / len(original_images)
# print(keypoint_accuracy)

processed_images = processed_images.reshape(-1, 512, 512, 3)

# Adatok és címkék összekeverése
processed_images, labels = shuffle(processed_images, labels, random_state=42)

X_train, X_test, y_train, y_test, image_paths_train, image_paths_test = train_test_split(
    processed_images, labels, image_paths, test_size=0.2, random_state=42, stratify=labels
)

# Adatbővítés
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Címkék átalakítása one-hot encoding formátumra
y_train = to_categorical(y_train, num_classes=len(config.LOGO_PATHS_WITH_TEMPLATES))
y_test = to_categorical(y_test, num_classes=len(config.LOGO_PATHS_WITH_TEMPLATES))

# Modell építése és betanítása
model = build_model((512, 512, 3))
loss, accuracy = train_model(model, X_train, y_train, X_test, y_test, epochs=20)

print(f'Teszt veszteség: {loss}')
print(f'Teszt pontosság: {accuracy}')

# label_dict = { 0: 'Honda', 1: 'Apple', 2: 'Nike', 3: 'Peugeot' }

def turn_cv_img_to_gui(cv_image):
    cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)

    # Convert the OpenCV image to a PIL Image
    pil_image = Image.fromarray(cv_image_rgb)

    # Convert the PIL Image to an ImageTk object
    tk_image = ImageTk.PhotoImage(pil_image)
    return tk_image

def show_random_image():
    # Select a random image from the test set
    idx = random.randint(0, len(X_test) - 1)
    img_path = image_paths_test[idx]

    # Load the original image from the file path
    original_image = cv.imread(img_path)

    original_image = resize_with_aspect_ratio(original_image, 256)

    image_tk = turn_cv_img_to_gui(original_image)

    # Get prediction for the image
    prediction, predicted_label = predict_image(model, X_test[idx])

    # Update CNN prediction label and image
    cnn_image_label.config(image=image_tk)
    cnn_image_label.image = image_tk
    cnn_result_label.config(text=f"CNN Prediction: {label_dict[predicted_label]} ({prediction[0][predicted_label]:.2f})")

    # For demonstration, use the same image for keypoint prediction
    best_logo, matched_image = predict_with_keypoint(original_image)
    keypoint_image_tk = turn_cv_img_to_gui(matched_image)

    keypoint_image_label.config(image=keypoint_image_tk)
    keypoint_image_label.image = keypoint_image_tk
    keypoint_result_label.config(text=f"Keypoint Prediction: {best_logo}")

# Create GUI
root = tk.Tk()
root.title("Test Set Predictions")

# Set window size
window_width = 1400
window_height = 800
root.geometry(f"{window_width}x{window_height}")

# Create and place the "Show random image" button
show_button = tk.Button(root, text="Show random image", command=show_random_image)
show_button.grid(row=0, column=0, columnspan=2, pady=10)

# Create and place CNN prediction image and label
cnn_image_label = tk.Label(root)
cnn_image_label.grid(row=1, column=0, padx=10, pady=10)
cnn_result_label = tk.Label(root, text="CNN Prediction: ")
cnn_result_label.grid(row=2, column=0, padx=10, pady=10)

# Create and place keypoint prediction image and label
keypoint_image_label = tk.Label(root)
keypoint_image_label.grid(row=1, column=1, padx=10, pady=10)
keypoint_result_label = tk.Label(root, text="Keypoint Prediction: ")
keypoint_result_label.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()