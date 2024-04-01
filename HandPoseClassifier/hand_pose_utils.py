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
        """Get hand landmarks from a single image."""
        # Convert the BGR image to RGB and process it with MediaPipe Hands.
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.multi_hand_landmarks

    def draw_hand_landmarks(self, image, landmarks):
        """Draw hand landmarks on an image."""
        image_with_landmarks = np.copy(image)
        # Draw hand landmarks of each hand.
        if landmarks:
            for hand_landmarks in landmarks:
                self.mp_drawing.draw_landmarks(
                    image_with_landmarks, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return image_with_landmarks

    def camera_feed(self):
        """Display webcam feed with hand landmarks. Press 'q' to quit. Press 's' to save the image with landmarks. Press 'd' to save the image without landmarks."""
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
        """Calculate the area and perimeter of the hand shape using convex hull."""
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
        """Calculate the angle between thumb and index finger."""
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
        """Calculate distance between thumb tip and index finger tip."""
        if not landmarks:
            return None

        # Calculate distance between thumb tip and index finger tip.
        thumb_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        return distance

    def calculate_palm_ratio(self, landmarks):
        """Calculate palm width, height, and ratio."""
        if not landmarks:
            return None, None, None

        # Get palm landmarks (wrist and base of fingers).
        palm_landmarks = [landmarks[0].landmark[i] for i in range(0, 21, 4)]

        # Calculate palm width (distance between wrist landmarks).
        palm_width = np.sqrt(
            (palm_landmarks[0].x - palm_landmarks[1].x) ** 2
            + (palm_landmarks[0].y - palm_landmarks[1].y) ** 2
        )

        # Calculate palm height (distance between base of fingers).
        palm_height = np.sqrt(
            (palm_landmarks[2].x - palm_landmarks[3].x) ** 2
            + (palm_landmarks[2].y - palm_landmarks[3].y) ** 2
        )

        # Calculate ratios.
        width_height_ratio = palm_width / palm_height

        return palm_width, palm_height, width_height_ratio

    def calculate_finger_curvature(self, landmarks):
        """Calculate curvature of each finger."""
        if not landmarks:
            return None

        # Define finger joints for curvature calculation.
        joints = [
            [self.mp_hands.HandLandmark.WRIST, self.mp_hands.HandLandmark.THUMB_CMC],
            [
                self.mp_hands.HandLandmark.THUMB_CMC,
                self.mp_hands.HandLandmark.THUMB_MCP,
            ],
            [self.mp_hands.HandLandmark.THUMB_MCP, self.mp_hands.HandLandmark.THUMB_IP],
            [self.mp_hands.HandLandmark.THUMB_IP, self.mp_hands.HandLandmark.THUMB_TIP],
            [
                self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            ],
            # Add other fingers as needed...
        ]

        # Calculate angles for each finger joint.
        finger_curvatures = []
        for joint in joints:
            start_point = landmarks[0].landmark[joint[0]]
            mid_point = landmarks[0].landmark[joint[1]]
            end_point = landmarks[0].landmark[joint[1] + 1]
            # Calculate curvature using the angle between vectors.
            vector1 = np.array(
                [start_point.x - mid_point.x, start_point.y - mid_point.y]
            )
            vector2 = np.array([end_point.x - mid_point.x, end_point.y - mid_point.y])
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
            angle_deg = np.degrees(angle_rad)
            finger_curvatures.append(angle_deg)

        return finger_curvatures

    def calculate_palm_center(self, landmarks):
        """Calculate the center of the palm."""
        if not landmarks:
            return None, None

        # Get palm landmarks (wrist and base of fingers).
        palm_landmarks = [landmarks[0].landmark[i] for i in range(0, 21, 4)]

        # Calculate center of palm as the midpoint between wrist landmarks.
        center_x = (palm_landmarks[0].x + palm_landmarks[1].x) / 2
        center_y = (palm_landmarks[0].y + palm_landmarks[1].y) / 2

        return center_x, center_y

    def calculate_finger_width_ratios(self, landmarks):
        """Calculate width ratios of fingers."""
        if not landmarks:
            return None

        # Get palm landmarks (wrist and base of fingers).
        palm_landmarks = [landmarks[0].landmark[i] for i in range(0, 21, 4)]

        # Calculate finger widths.
        finger_widths = []
        for i in range(1, 5):  # Assuming there are 5 fingers including thumb
            finger_width = np.sqrt(
                (palm_landmarks[0].x - landmarks[0].landmark[i * 4].x) ** 2
                + (palm_landmarks[0].y - landmarks[0].landmark[i * 4].y) ** 2
            )
            finger_widths.append(finger_width)

        # Calculate ratios.
        width_ratios = [
            finger_widths[i] / finger_widths[0] for i in range(1, len(finger_widths))
        ]

        return width_ratios

    def extract_features(self, image):
        """Extract hand pose features from a single image."""
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

        # Calculate palm width, height, and ratio features.
        palm_width, palm_height, width_height_ratio = self.calculate_palm_ratio(
            landmarks
        )

        # Calculate finger curvature features.
        finger_curvatures = self.calculate_finger_curvature(landmarks)

        # Calculate palm center features.
        palm_center_x, palm_center_y = self.calculate_palm_center(landmarks)

        # Calculate finger width ratio features.
        finger_width_ratios = self.calculate_finger_width_ratios(landmarks)

        # Concatenate all features into one array.
        features = np.array(
            [
                hand_area,
                hand_perimeter,
                thumb_index_angle_deg,
                thumb_index_distance,
                palm_width,
                palm_height,
                width_height_ratio,
                palm_center_x,
                palm_center_y,
                *finger_curvatures,
                *finger_width_ratios,
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
