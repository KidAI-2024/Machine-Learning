import pytest
from handlers.body_pose_handlers import *
from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
from server_utils import Req, Res
import base64
import cv2

class TestBodyPoseHandlers:
    def test_predict_without_training(self, black_image):
        """
        tests the predict function without training the model
        should return a prediction of None
        """
        frame_bytes = black_image
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "predict_body_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = predict_body_pose(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "prediction": "None"}
        print("Predict without training test passed")

    def test_predict_with_training(self, black_image, mocker):
        """
        tests the predict function without training the model
        should return a prediction of 0
        """
        mocker.patch.object(BodyPoseClassifier, "predict").return_value = 0

        frame_bytes = black_image
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "predict_body_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = predict_body_pose(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "prediction": 0}
        print("Predict with training test passed")

    def test_preprocess_body_pose(self, black_image):
        """
        tests the preprocess_body_pose function
        should return a preprocessed image
        """
        frame_bytes = black_image
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "preprocess_body_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = preprocess_body_pose(req, res)

        # Assert that the result is as expected
        assert "preprocessed_image" in result
        print("Preprocess test passed")

    def test_load_model_correct_path(self, mocker):
        # mock body_pose_classifier.load(model_path)
        """
        mock the load function avoid raising an exception for not finding the path
        """
        mocker.patch.object(BodyPoseClassifier, "load")
        path = "C:/Users/username/Documents/Projects/BodyPoseClassifier/models"
        saved_model_name = "body_pose_model.pkl"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        body_pose_classifier.selected_features = ["feature1", "feature2"]
        msg = {"path": path, "saved_model_name": saved_model_name, "model": model, "feature_extraction_type": feature_extraction_type, "features": "feature1,feature2"}
        event = "load_body_pose_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = load_body_pose_model(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "status": "success"}
        print("load model correct path test passed")

    def test_load_model_uncorrect_path(self, mocker):
        # mock body_pose_classifier.load(model_path)
        """
        mock the load function avoid raising an exception for not finding the path
        """
        mocker.patch.object(BodyPoseClassifier, "load").side_effect = FileNotFoundError
        path = "C:/Users/username/Documents/Projects/BodyPoseClassifier/models"
        saved_model_name = "body_pose_model.pkl"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        body_pose_classifier.selected_features = ["feature1", "feature2"]
        msg = {"path": path, "saved_model_name": saved_model_name, "model": model, "feature_extraction_type": feature_extraction_type, "features": "feature1,feature2"}
        event = "load_body_pose_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = load_body_pose_model(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "status": "failed"}
        print("load model uncorrect path test passed")


    def test_train_body_pose(self, mocker):
        """
        tests the train_body_pose function
        should return a status of success
        """
        # mock the body_pose_classifier.train function
        mocker.patch.object(BodyPoseClassifier, "train")
        # mock body_pose_classifier.preprocess function
        mocker.patch.object(BodyPoseClassifier, "preprocess")
        # mock body_pose_classifier.feature_importance_graph function
        mocker.patch.object(BodyPoseClassifier, "feature_importance_graph").return_value = "image_string"
        # mock the utils.read_data function
        mocker.patch("handlers.body_pose_handlers.utils.read_data").return_value = {"0": ["image1.png", "image2.png"], "1": ["image3.png", "image4.png"]}
        # mock the body_pose_classifier.save function
        mocker.patch.object(BodyPoseClassifier, "save")
        path = "C:/Users/username/Documents/Projects/BodyPoseClassifier/models"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        body_pose_classifier.selected_features = ["feature1", "feature2"]
        saved_model_name = "body_pose_model.pkl"
        feature_importance_graph = "image_string"
        msg = {"path": path, "model": model, "feature_extraction_type": feature_extraction_type, "features": "feature1,feature2"}
        event = "train_body_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = train_body_pose(req, res)
    
        # Assert that the result is as expected
        assert result == {"event": event, "status": "success", "saved_model_name": saved_model_name, "feature_importance_graph" : feature_importance_graph}
        print("Start train test passed")


@pytest.fixture
def black_image():
    image_rgb = cv2.cvtColor(cv2.imread("./Tests/images/black_image.png"), cv2.COLOR_BGR2RGB)
    # image should look like 
    image = base64.b64encode(image_rgb).decode("utf-8")
    return image