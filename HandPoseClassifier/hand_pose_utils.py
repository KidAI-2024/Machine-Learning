import cv2
import mediapipe as mp


class HandPoseUtils:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def get_hand_landmarks(self, image):
        # Convert the BGR image to RGB and process it with MediaPipe Hands.
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw hand landmarks of each hand.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return image

    def camera_feed(self):
        # Initialize the webcam.
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            # Capture a frame from the webcam.
            ret, frame = cap.read()

            if not ret:
                print("Unable to acquire webcam feed.")
                break

            # Get hand landmarks
            frame = self.get_hand_landmarks(frame)

            # Display the frame.
            cv2.imshow("MediaPipe Hands", frame)

            # Break the loop if 'q' is pressed.
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        # Close MediaPipe Hands.
        self.hands.close()

        # Release the webcam and close the OpenCV window.
        cap.release()
        cv2.destroyAllWindows()
