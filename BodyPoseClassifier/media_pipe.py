import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 


def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
    
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def extract_features(image):
    # Flip the image horizontally for natural (selfie-view) visualization.
    image = cv2.flip(image, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  image.shape
    
    # Resize the frame while keeping the aspect ratio.
    image = cv2.resize(image, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    image, landmarks = detectPose(image, pose_video, display = False)


# make a function that takes a list of images and returns the landmarks
def extract_features_from_images(images):
    # Create a list to store the landmarks of each image.
    landmarks_list = []
    
    # Iterate over the images.
    for image in images:
        
        # Perform Pose landmark detection.
        _ , landmarks = detectPose(image, pose_video, display = False)
        print("image : ", landmarks)
        # Append the landmarks into the list.
        landmarks_list.append(landmarks)
    
    # Return the list of landmarks.
    return landmarks_list