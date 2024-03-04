import logging
import utils
import cv2
import time


class EventHandlers:
    def __init__(
        self,
        hand_pose_classifier,
        image_classifier,
        body_pose_classifier,
        audio_classifier,
    ):
        self.hand_pose_classifier = hand_pose_classifier
        self.image_classifier = image_classifier
        self.body_pose_classifier = body_pose_classifier
        self.audio_classifier = audio_classifier

        self.EVENTS = {
            "predict_frame": self.predict_frame,
            "start_body_pose_train": self.start_body_pose_train,
            "predict_hand_pose": self.predict_hand_pose,
        }

    def handle_event(self, event, message_obj):
        if event in self.EVENTS:
            if event == "start_body_pose_train":
                data_path = message_obj["path"]
                self.EVENTS[event](data_path)
            elif event == "predict_frame":
                frame_bytes = message_obj["frame"]
                width_str = message_obj["width"]
                height_str = message_obj["height"]
                try:
                    width = int(width_str)
                except ValueError:
                    width = 320
                    logging.error(f"Invalid width: {width_str}")
                try:
                    height = int(height_str)
                except ValueError:
                    height = 180
                    logging.error(f"Invalid height: {height_str}")
                # Convert the bytes to an image
                image = utils.bytes_to_image(frame_bytes, (height, width, 3))
                self.EVENTS[event](image)
        else:
            logging.error(f"Event '{event}' not found")

    # ------- Event handlers -------

    # --- General ---
    def predict_frame(self, image):
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
        pass

    # --- Body Pose Classifier ---
    def start_body_pose_train(self, path):
        # training_data is map {"Class Number(first character in the folder name)" : [images]}
        training_data = utils.read_data(path)
        for key in training_data:
            print(f"Training for class {key} length: {len(training_data[key])}")
        return 0

    # --- Hand Pose Classifier ---
    def predict_hand_pose(self, image):
        preprocessed_img = self.hand_pose_classifier.preprocess(image)
        cv2.imwrite(f"./frames_test/frame_{time.time()}.png", preprocessed_img)
        pass
