import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os

class BodyPoseUtils:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils 

    def get_body_landmarks(self, image):
        """Get body landmarks from a single image."""
        # Convert the BGR image to RGB and process it with MediaPipe Body.
        try:
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return results.pose_landmarks
        except Exception as e:
            print(
                "Error while processing image in body_pose_utils::get_body_landmarks:",
                e,
            )
            return None
        
    def draw_body_landmarks(self, image, landmarks, copy=False):
        """Draw body landmarks on an image."""
        # Make a copy of the image if copy is True.
        image_with_landmarks = np.copy(image) if copy else image
        # Draw body landmarks of each body.
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image_with_landmarks,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
        return image_with_landmarks
    
    def camera_feed(self):
        """Display webcam feed with body landmarks. Press 'q' to quit. Press 's' to save the image with landmarks. Press 'd' to save the image without landmarks."""
        # Initialize the webcam.
        cap = cv2.VideoCapture(0)
        # Set the resolution to 320x180
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

        while cap.isOpened():
            # Capture a frame from the webcam.
            ret, frame = cap.read()

            if not ret:
                print("Unable to acquire webcam feed.")
                break

            # Get body landmarks
            landmarks = self.get_body_landmarks(frame)
            frame_with_landmarks = self.draw_body_landmarks(frame, landmarks)

            # Display the frame.
            cv2.imshow("MediaPipe Body", frame_with_landmarks)
            # Wait for a key press
            key = cv2.waitKey(5)
            # Break the loop if 'q' is pressed.
            if key & 0xFF == ord("q"):
                break
            # save the image with landmarks if 's' is pressed.
            if key & 0xFF == ord("s"):
                body_path = os.path.join("test_images", f"body_image_{time.time()}.jpg")
                cv2.imwrite(body_path, frame_with_landmarks)
                print(f"Body image saved at {body_path}")
            # save the image without landmarks if 'd' is pressed.
            if key & 0xFF == ord("d"):
                body_path = os.path.join("test_images", f"body_image_{time.time()}.jpg")
                cv2.imwrite(body_path, frame)
                print(f"Body image saved at {body_path}")

        # Close MediaPipe Body.
        self.pose.close()

        # Release the webcam and close the OpenCV window.
        cap.release()
        cv2.destroyAllWindows()

    
    def calculate_body_shape(self, landmarks):
        """Calculate the area and perimeter of the body shape using convex hull."""
        if not landmarks or not landmarks[0].landmark:
            print("No landmarks detected or landmarks contain no data.")
            return None, None

        points = np.array(
            [(lm.x, lm.y) for lm in landmarks[0].landmark], dtype=np.float32
        )
        if len(points) < 3:
            print("Insufficient landmarks to compute convex hull.")
            return None, None

        try:
            hull = cv2.convexHull(points)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, closed=True)
            return area, perimeter
        except Exception as e:
            print("Error while computing convex hull:", e)
            return None, None


    def angle_between_three_points(self, pointA, pointB, pointC):
        # Vectors AB and BC
        AB = np.array([pointA.x - pointB.x, pointA.y - pointB.y, pointA.z - pointB.z])
        BC = np.array([pointC.x - pointB.x, pointC.y - pointB.y, pointC.z - pointB.z])
        # Normalize vectors
        AB_normalized = AB / np.linalg.norm(AB)
        BC_normalized = BC / np.linalg.norm(BC)
        # Calculate the dot product
        dot_product = np.dot(AB_normalized, BC_normalized)
        # Calculate the angle in radians and then convert to degrees
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    
    def calculate_symmetry_score(self, landmarks):
        '''Calculate the symmetry score of the body pose.'''
        left_indices = [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                        self.mp_pose.PoseLandmark.LEFT_WRIST.value,
                        self.mp_pose.PoseLandmark.LEFT_HIP.value,
                        self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                        self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_indices = [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                        self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                        self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                        self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                        self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_landmarks = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in left_indices])
        right_landmarks = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in right_indices])
        
        # Calculate distances between corresponding left and right landmarks
        distances = np.linalg.norm(left_landmarks - right_landmarks, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Symmetry score
        symmetry_score = avg_distance / max_distance if max_distance != 0 else 0
        
        return symmetry_score

    
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array([point1.x, point1.y, point1.z]) - np.array([point2.x, point2.y, point2.z]))
    
    def extract_features(self, image):
        """Extract hand pose features from a single image."""
        # Get hand landmarks.
        landmarks = self.get_body_landmarks(image).landmark
        # Key points
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate distances
        l_dist_shoulder_to_elbow = self.calculate_distance(l_shoulder, l_elbow)
        l_dist_elbow_to_wrist = self.calculate_distance(l_elbow, l_wrist)
        l_dist_shoulder_to_wrist = self.calculate_distance(l_shoulder, l_wrist)
        r_dist_shoulder_to_elbow = self.calculate_distance(r_shoulder, r_elbow)
        r_dist_elbow_to_wrist = self.calculate_distance(r_elbow, r_wrist)
        r_dist_shoulder_to_wrist = self.calculate_distance(r_shoulder, r_wrist)

        # Calculate angles between specific landmarks
        l_angle_shoulder_elbow_wrist = self.angle_between_three_points(l_shoulder, l_elbow, l_wrist)
        l_angle_elbow_shoulder_hip = self.angle_between_three_points(l_elbow, l_shoulder, l_hip)
        r_angle_shoulder_elbow_wrist = self.angle_between_three_points(r_shoulder, r_elbow, r_wrist)
        r_angle_elbow_shoulder_hip = self.angle_between_three_points(r_elbow, r_shoulder, r_hip)
        
        # Calculate symmetry
        symmetry_score = self.calculate_symmetry_score(landmarks)

        # Concatenate all features into one array.
        features = np.array(
            [
                l_angle_elbow_shoulder_hip, 
                l_angle_shoulder_elbow_wrist, 
                l_dist_shoulder_to_elbow, 
                l_dist_elbow_to_wrist, 
                l_dist_shoulder_to_wrist, 
                r_angle_shoulder_elbow_wrist, 
                r_angle_elbow_shoulder_hip, 
                r_dist_shoulder_to_elbow, 
                r_dist_elbow_to_wrist, 
                r_dist_shoulder_to_wrist, 
                symmetry_score
            ]
        )
        return features
    
    
    def get_training_features(self, training_data):
        """Extract hand pose features from the dictionary of training images."""
        features_map = {}
        for class_name, images in training_data.items():
            features_list = []
            for image in images:
                features = self.extract_features(image)
                features_list.append(features)
            features_map[class_name] = features_list
        return features_map
    


if __name__ == "__main__":
    body_pose_utils = BodyPoseUtils()
    # -- camera feed --
    # body_pose_utils.camera_feed()

    # -- open image --
    image_path = os.path.join("test_images", "body_image_1709731770.3709466.jpg")
    image = cv2.imread(image_path)

    ## -- display image --
    # while True:
    #     # Display the frame.
    #     cv2.imshow("MediaPipe Hands", image)
    #     # Wait for a key press
    #     key = cv2.waitKey(5)
    #     # Break the loop if 'q' is pressed.
    #     if key & 0xFF == ord("q"):
    #         break

    # -- feature extraction --
    features = body_pose_utils.extract_features(image)
    print("Features:", features)
