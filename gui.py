import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random

class ImageLoaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("4 Random Images Viewer")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")  # Sötét háttér

        # Címke
        self.title_label = tk.Label(self.root, text="4 Random Images Viewer", font=("Arial", 20, "bold"), bg="#2c3e50", fg="white")
        self.title_label.pack(pady=20)

        # Frame a képek számára
        self.image_frame = tk.Frame(self.root, bg="#34495e")
        self.image_frame.pack(pady=20)

        # Helyek a képeknek
        self.image_labels = [tk.Label(self.image_frame, bg="#34495e") for _ in range(4)]
        for i, lbl in enumerate(self.image_labels):
            lbl.grid(row=0, column=i, padx=10, pady=10)

        # Gombok
        self.button_frame = tk.Frame(self.root, bg="#2c3e50")
        self.button_frame.pack(pady=20)

        # Betöltés gomb
        self.load_button = tk.Button(self.button_frame, text="Load Images", command=self.load_images, bg="#1abc9c", fg="white", font=("Arial", 12, "bold"), width=15)
        self.load_button.pack(side=tk.LEFT, padx=10)

        # Logó keresés gomb
        self.find_button = tk.Button(self.button_frame, text="Find Logos", command=self.find_logos, bg="#1abc9c", fg="white", font=("Arial", 12, "bold"), width=15)
        self.find_button.pack(side=tk.LEFT, padx=10)

        # Képek törlés gomb
        self.clear_button = tk.Button(self.button_frame, text="Clear Images", command=self.clear_images, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), width=15)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # Logók mappái
        self.logo_folders = {
            "Nike": "assets/logos/nike",
            "Apple": "assets/logos/apple",
            "Honda": "assets/logos/honda",
            "Peugeot": "assets/logos/peugeot",
        }

    def load_images(self):
        selected_images = []

        # Mindegyik mappából véletlenszerűen kiválasztunk egy képet
        for brand, folder_path in self.logo_folders.items():
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                selected_images.append(None)
                continue

            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"No images found in {brand} folder.")
                selected_images.append(None)
                continue

            random_image = random.choice(images)
            selected_images.append(os.path.join(folder_path, random_image))

        # Képek betöltése és megjelenítése
        for i, img_path in enumerate(selected_images):
            if img_path is None:
                self.image_labels[i].config(image=None, text=f"No Image for {list(self.logo_folders.keys())[i]}", fg="white")
                self.image_labels[i].image = None
                continue

            img = Image.open(img_path).resize((150, 150))  # Átméretezés
            img_tk = ImageTk.PhotoImage(img)

            self.image_labels[i].config(image=img_tk, text="")
            self.image_labels[i].image = img_tk  # Tárolni kell a referenciát!

    def clear_images(self):
        # Törli a képeket a GUI-ból
        for lbl in self.image_labels:
            lbl.config(image=None, text="No Image", bg="#34495e", fg="white")
            lbl.image = None

    def find_logos(self):
        print("Finding logos...")  
        # TODO  a 4 random  képen meg kell keresni melyik logó szerepel és jelezni valahogy.
        # ötlet kulcspontok lementése , keresés  a 4 képen => legjobb logó.

# Indítás
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderGUI(root)
    root.mainloop()
