import os
import cv2

# returns map {"Class Number(first character in the folder name)" : [images]}
def read_data(path):
    try:
        project_path = os.path.join("../","Engine", path)
        # Create empty map {"Class Number(first character in the folder name)" : [images]}
        training_data = {} 
        # Loop over folders in the specified path
        for folder in os.listdir(project_path):
            folder_path = os.path.join(project_path, folder)
            class_name = folder[0]
            if os.path.isdir(folder_path):
                images = []
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.png'):
                        image_path = os.path.join(folder_path, file_name)
                        image = cv2.imread(image_path)
                        images.append(image)
                training_data[class_name] = images
    except:
        # Error reading data from path (wrong path)
        print("Error reading data from path")
        return {}
    return training_data