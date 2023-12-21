import os

from PIL import Image
import numpy as np


def normalize_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            image_array = np.array(image)

            normalized_image = image_array.astype(np.float32) / 255.0

            normalized_image = Image.fromarray((normalized_image * 255).astype(np.uint8))

            output_path = os.path.join(output_folder, filename)
            normalized_image.save(output_path)


input_folder = "../dataset/cardboard"
output_folder = "../dataset/normalized/cardboard"

normalize_images(input_folder, output_folder)
