import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, display=True):
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Setting up the Pose function.
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB));plt.title("Output Image");plt.axis('off')
        plt.show()
        
    return output_image, landmarks
    
def get_landmarks(image, display = False):
    # Flip the image horizontally for natural (selfie-view) visualization.
    image = cv2.flip(image, 1)

    # Check if the image is successfully read
    if image is None:
        print(f"Error: Unable to read image")
        return
    # Get the width and height of the frame
    frame_height, frame_width, _ =  image.shape
    
    # Resize the frame while keeping the aspect ratio.
    image = cv2.resize(image, (int(frame_width * (640 / frame_height)), 640))
    # Perform Pose landmark detection.
    image, landmarks = detectPose(image, display)

    return image, landmarks

# make a function that takes a list of images and returns the landmarks
def get_landmarks_from_images(images, display = False):
    # Create a list to store the landmarks of each image.
    landmarks_list = []
    images_list = []
    
    # Iterate over the images.
    for image in images:
        # Perform Pose landmark detection.
        image, landmarks = detectPose(image, display)
        # Append the landmarks into the list.
        landmarks_list.append(landmarks)
        images_list.append(image)
    
    # Return the list of landmarks.
    return images_list, landmarks_list

def relative_positions_features(landmarks):
    # Convert landmarks to NumPy array for easier computation
    landmarks = np.array(landmarks)
    
    # Calculate center of mass
    center_x = np.mean(landmarks[:, 0])
    center_y = np.mean(landmarks[:, 1])
    
    # Calculate distances between specific landmarks
    dist_shoulder_to_elbow = np.linalg.norm(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    dist_elbow_to_wrist = np.linalg.norm(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value] - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    dist_shoulder_to_wrist = np.linalg.norm(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Calculate angles between specific landmarks
    angle_shoulder_elbow_wrist = angle_between_three_points(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Calculate symmetry
    symmetry_score = calculate_symmetry_score(landmarks)
    # Construct a dictionary with feature names and their values
    features = {
        "Center of mass X": center_x,
        "Center of mass Y": center_y,
        "Distance shoulder to elbow": dist_shoulder_to_elbow,
        "Distance elbow to wrist": dist_elbow_to_wrist,
        "Distance shoulder to wrist": dist_shoulder_to_wrist,
        "Angle shoulder-elbow-wrist": angle_shoulder_elbow_wrist,
        "Symmetry score": symmetry_score
    }
    # Other feature extraction techniques can be added here
    
    return features # center_x, center_y, dist_shoulder_to_elbow, dist_elbow_to_wrist, dist_shoulder_to_wrist, angle_shoulder_elbow_wrist, symmetry_score

def angle_between_three_points(p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / norm_product)
    return np.degrees(angle_rad)

import numpy as np

def calculate_symmetry_score(landmarks):
    left_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value]

    right_indices = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                     mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                     mp_pose.PoseLandmark.RIGHT_WRIST.value,
                     mp_pose.PoseLandmark.RIGHT_HIP.value,
                     mp_pose.PoseLandmark.RIGHT_KNEE.value,
                     mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    left_landmarks = np.array([landmarks[i] for i in left_indices])
    right_landmarks = np.array([landmarks[i] for i in right_indices])
    avg_distance = np.mean(np.linalg.norm(left_landmarks - right_landmarks, axis=1))
    max_distance = np.max(np.linalg.norm(left_landmarks - right_landmarks, axis=1))
    return avg_distance / max_distance


# Plotting function
def plot_images(img_original, img_with_landmarks):
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Plot image with landmarks
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
    plt.title("Image with Landmarks")
    plt.axis("off")
    
    # Show the plot
    plt.show()


img = cv2.imread("test.jpg")
image, landmarks = get_landmarks(img)
features = relative_positions_features(landmarks)
for feature_name, feature_value in features.items():
    print(f"{feature_name}: {feature_value}")

# plot the image before and after in subplots
plot_images(img, image)