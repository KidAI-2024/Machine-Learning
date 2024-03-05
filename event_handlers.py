import logging
import utils
import time
import cv2

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
        }
    
    def handle_event(self, event, message_obj):
        res = ""
        if event in self.EVENTS:
            if event == "start_body_pose_train":
                data_path = message_obj["path"]
                res = self.EVENTS[event](data_path)
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
                res = self.EVENTS[event](image)
        else:
            res = f"Event '{event}' not found"
            logging.error(f"Event '{event}' not found")

        return res

    # ------- Event handlers -------

    # --- General ---
    def predict_frame(self, image):
        # cv2.imwrite(f"frame_{time.time()}.png", image)
        prediction = self.body_pose_classifier.predict(image)
        return prediction
        
    def start_body_pose_train(self, path):
        # training_data is map {"Class Number(first character in the folder name)" : [images]}
        print("Reading data...")
        training_data = utils.read_data(path)
        print("Getting keypoints...")
        preprocessed_data, landmarks_map = self.body_pose_classifier.preprocess(training_data)
        print("Extracting features...")
        features_map = self.body_pose_classifier.get_training_features(landmarks_map)
        print("Training...")
        self.body_pose_classifier.train(features_map)
        # print("Saving model...")
        # self.body_pose_classifier.save_model(path)
        print("Training completed")
        return "Training completed"
    
