import cv2
import mediapipe as mp
import numpy as np
import os
import time


class HandPoseUtils:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def get_hand_landmarks(self, image):
        # Convert the BGR image to RGB and process it with MediaPipe Hands.
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.multi_hand_landmarks

    def draw_hand_landmarks(self, image, landmarks):
        image_with_landmarks = np.copy(image)
        # Draw hand landmarks of each hand.
        if landmarks:
            for hand_landmarks in landmarks:
                self.mp_drawing.draw_landmarks(
                    image_with_landmarks, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return image_with_landmarks

    def camera_feed(self):
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

            # Get hand landmarks
            landmarks = self.get_hand_landmarks(frame)
            frame_with_landmarks = self.draw_hand_landmarks(frame, landmarks)

            # Display the frame.
            cv2.imshow("MediaPipe Hands", frame_with_landmarks)
            # Wait for a key press
            key = cv2.waitKey(5)
            # Break the loop if 'q' is pressed.
            if key & 0xFF == ord("q"):
                break
            # save the image with landmarks if 's' is pressed.
            if key & 0xFF == ord("s"):
                hand_path = os.path.join("test_images", f"hand_image_{time.time()}.jpg")
                cv2.imwrite(hand_path, frame_with_landmarks)
                print(f"Hand image saved at {hand_path}")
            # save the image without landmarks if 'd' is pressed.
            if key & 0xFF == ord("d"):
                hand_path = os.path.join("test_images", f"hand_image_{time.time()}.jpg")
                cv2.imwrite(hand_path, frame)
                print(f"Hand image saved at {hand_path}")

        # Close MediaPipe Hands.
        self.hands.close()

        # Release the webcam and close the OpenCV window.
        cap.release()
        cv2.destroyAllWindows()

    def calculate_hand_shape(self, landmarks):
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

    def calculate_finger_angles(self, landmarks):
        # Calculate angles between finger joints.
        if not landmarks:
            return None

        # Example: Calculate angle between thumb and index finger.
        thumb_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Calculate angles using dot product and arccosine.
        vector1 = np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
        vector2 = np.array([middle_tip.x - index_tip.x, middle_tip.y - index_tip.y])
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_finger_spread(self, landmarks):
        # Calculate distance between fingertips of adjacent fingers.
        if not landmarks:
            return None

        # Example: Calculate distance between thumb tip and index finger tip.
        thumb_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        return distance

    # Add more methods for other features (e.g., palm features, finger spreads).
    def extract_features(self, image):
        # Get hand landmarks.
        landmarks = self.get_hand_landmarks(image)
        # Draw hand landmarks on the image.
        # image_with_landmarks = self.draw_hand_landmarks(image, landmarks)
        # Calculate hand shape features.
        hand_area, hand_perimeter = self.calculate_hand_shape(landmarks)
        # Calculate finger angle features.
        thumb_index_angle_deg = self.calculate_finger_angles(landmarks)
        # Calculate finger spread features.
        thumb_index_distance = self.calculate_finger_spread(landmarks)
        # print(f"Hand Shape Features: Area = {hand_area}, Perimeter = {hand_perimeter}")
        # print("Finger Angle Features:", thumb_index_angle_deg)
        # print("Finger Spread Features:", thumb_index_distance)
        features = np.array(
            [hand_area, hand_perimeter, thumb_index_angle_deg, thumb_index_distance]
        )
        return features


if __name__ == "__main__":
    hand_pose_utils = HandPoseUtils()
    # -- camera feed --
    # hand_pose_utils.camera_feed()

    # -- open image --
    image_path = os.path.join("test_images", "hand_image_1709731770.3709466.jpg")
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
    features = hand_pose_utils.extract_features(image)
    print("Features:", features)
