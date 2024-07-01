import os
import cv2
import base64
import numpy as np
import os
import torch





# returns map {"Class Number(first character in the folder name)" : [images]}
def read_data(path):
    try:
        project_path = os.path.join("..", "Engine", path)
        # Create empty map {"Class Number(first character in the folder name)" : [images]}
        training_data = {}
        # Loop over folders in the specified path
        for folder in os.listdir(project_path):
            folder_path = os.path.join(project_path, folder)
            class_name = folder[0]
            if os.path.isdir(folder_path):
                images = []
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".png"):
                        image_path = os.path.join(folder_path, file_name)
                        image = cv2.imread(image_path)
                        images.append(image)
                training_data[class_name] = images
    except:
        # Error reading data from path (wrong path)
        print("Error reading data from path")
        return {}
    return training_data


def b64string_to_image(frame_bytes, shape):
    # Get the image data
    image_data = base64.b64decode(frame_bytes)
    color_array = np.frombuffer(image_data, dtype=np.uint8)
    image = color_array.reshape(shape)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip the image vertically
    image = cv2.flip(image, 0)
    return image


def b64string_to_image_float(frame_bytes, shape):
    image = b64string_to_image(frame_bytes, shape)
    # Convert the image to float
    image = image.astype(np.float32)
    return image


def image_to_b64string(image):
    # # Flip the image vertically
    # image = cv2.flip(image, 0)
    # # Convert the image from RGB to BGR
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Encode the image to bytes
    _, buffer = cv2.imencode(".jpg", image)
    # encode to base64 string
    image_bytes = base64.b64encode(buffer)
    image_str = image_bytes.decode("utf-8")
    return image_str
