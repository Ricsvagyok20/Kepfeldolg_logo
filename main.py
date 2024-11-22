import config

from chamfer_matching.preprocess import process_images
from chamfer_matching.resize import resize_with_aspect_ratio

# Fő program
image_paths = []
processed_images = []

# Iterate through each logo path and its corresponding template
for logo_group in config.LOGO_PATHS_WITH_TEMPLATES:
    logo_paths = logo_group['logo_paths']
    image_paths.extend(logo_paths)
    template_path = logo_group['template_path']

    # Process images for the current logo group
    group_processed_images = process_images(logo_paths, template_path)
    processed_images.extend(group_processed_images)

import numpy as np
import random
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from chamfer_matching.cnn_model import build_model, train_model, predict_image


# Címkék generálása
labels = np.array([0 if i < len(processed_images) // 2 else 1 for i in range(len(processed_images))])

# Adatok átalakítása a CNN számára
processed_images = processed_images.reshape(-1, 512, 512, 3)

# Adatok szétválasztása tanító és teszt adatokra
X_train, X_test, y_train, y_test, image_paths_train, image_paths_test = train_test_split(
    processed_images, labels, image_paths, test_size=0.2, random_state=42
)
# Modell építése és betanítása
model = build_model((512, 512, 3))
loss, accuracy = train_model(model, X_train, y_train, X_test, y_test)

print(f'Teszt veszteség: {loss}')
print(f'Teszt pontosság: {accuracy}')

# Create GUI
root = tk.Tk()
root.title("Test Set Predictions")

# Display all test set images with predictions
for i, img_path in enumerate(image_paths_test):
    # Load the original image from the file path
    original_image = Image.open(img_path)
    # original_image = original_image.resize((128, 128), Image.Resampling.LANCZOS)
    original_image = resize_with_aspect_ratio(original_image, 128)
    image_tk = ImageTk.PhotoImage(original_image)

    # Get prediction for the image
    prediction = predict_image(model, X_test[i])

    # Create and place image and prediction label
    image_label = tk.Label(root, image=image_tk)
    image_label.image = image_tk
    image_label.grid(row=i // 5 * 2, column=i % 5)  # Arrange in a grid (5 images per row)

    result_label = tk.Label(root, text=f"Prediction: {prediction[0][0]:.2f}")
    result_label.grid(row=i // 5 * 2 + 1, column=i % 5)  # Place below each image

root.mainloop()