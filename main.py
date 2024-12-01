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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keypoint_detection.keypointDetection import predict_with_keypoint

# Fő program
original_images = []
image_paths = []
processed_images = []

# for i in glob.glob('processed_images/*.jpg'):
#     image = cv.imread(i)
#     image_paths.append(i)
#
#     original_images.append(image)
#
# normalized_images = np.array(original_images)
#
# processed_images = normalized_images / 255.0

# Iterate through each logo path and its corresponding template
for logo_group in config.LOGO_PATHS_WITH_TEMPLATES:
    logo_paths = logo_group['logo_paths']
    image_paths.extend(logo_paths)
    template_path = logo_group['template_path']

    # Process images for the current logo group
    group_processed_images, original_image_group = process_images(logo_paths, template_path)
    original_images.extend(original_image_group)
    processed_images.append(group_processed_images)

processed_images = np.concatenate(processed_images, axis=0)

# Címkék generálása
labels = []
for i, logo_group in enumerate(config.LOGO_PATHS_WITH_TEMPLATES):
    logo_paths = logo_group['logo_paths']
    labels.extend([i] * len(logo_paths))

labels = np.array(labels)

label_dict = { 0: 'Honda', 1: 'Apple', 2: 'Nike', 3: 'Peugeot' }


processed_images = processed_images.reshape(-1, 512, 512, 3)

# Adatok és címkék összekeverése
processed_images, labels = shuffle(processed_images, labels, random_state=42)

X_train, X_test, y_train, y_test, image_paths_train, image_paths_test = train_test_split(
    processed_images, labels, image_paths, test_size=0.4, random_state=42, stratify=labels
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

print(f'CNN Teszt veszteség: {loss}')
print(f'CNN Teszt pontosság: {accuracy}')

def create_plots(predicted_labels, label_dict, cumulated_accuracy, is_cnn):
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    if is_cnn:
        numeric_labels = np.array(predicted_labels)
    else:
        numeric_labels = np.array([reverse_label_dict[logo.capitalize()] for logo in predicted_labels])

    cm = confusion_matrix([np.argmax(vec) for vec in y_test] if is_cnn else labels, numeric_labels)

    class_names = ["Apple", "Honda", "Nike", "Peugeot"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    type = 'CNN' if is_cnn else 'Keypoint'
    name = 'Confusion_Matrix_' + type + '.png'
    disp.ax_.set_title(type + ' confusion matrix')
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.bar(class_names, class_accuracies, color='skyblue')
    name = 'Accuracy_per_Class_' + type + '.png'
    percentage = f"{cumulated_accuracy * 100:.0f}%"
    plt.title('Cumulated accuracy: ' + str(percentage) + ' \n' + type + ' accuracy per Class')
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    for i, acc in enumerate(class_accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')

    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

cnn_predictions = []
for i in X_test:
    prediction, predicted_label = predict_image(model, i)
    cnn_predictions.append(predicted_label)

cnn_accuracy = 0
for i, label in enumerate(y_test):
    if np.argmax(label, axis=0) == cnn_predictions[i]:
        cnn_accuracy += 1
cnn_accuracy = cnn_accuracy / len(X_test)
# print(f'CNN predikció pontosság: {cnn_accuracy}')

# Keypoint detection accuracy mérés
labels_keypoint = []
for image in original_images:
    prediction, image_2 = predict_with_keypoint(image)
    labels_keypoint.append(prediction)

keypoint_accuracy = 0
for i, label in enumerate(labels):
    if label_dict[label].lower() == labels_keypoint[i]:
        keypoint_accuracy += 1
keypoint_accuracy = keypoint_accuracy / len(original_images)
# print(f'Kulcspont detektálás pontosság: {keypoint_accuracy}')


create_plots(cnn_predictions, label_dict, cnn_accuracy, True)
create_plots(labels_keypoint, label_dict, keypoint_accuracy, False)

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

# Create and place the "Show random image" button
show_button = tk.Button(root, text="Show random image", command=show_random_image)
show_button.grid(row=0, column=0, columnspan=3, pady=20)

root.mainloop()