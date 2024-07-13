import pytest
from handlers.hand_pose_handlers import *
from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
from server_utils import Req, Res


class TestHandPoseHandlers:
    def test_train_success(self, mocker):
        """
        tests the train_hand_pose function in case of success
        should return a status of success
        """
        mocker.patch.object(HandPoseClassifier, "preprocess")
        mocker.patch.object(HandPoseClassifier, "train")
        mocker.patch.object(
            HandPoseClassifier, "feature_importance_graph"
        ).return_value = "image_string"
        mocker.patch("handlers.hand_pose_handlers.utils.read_data").return_value = {
            "0": ["image1.png", "image2.png"],
            "1": ["image3.png", "image4.png"],
        }
        mocker.patch.object(HandPoseClassifier, "save")
        path = "path_to_model"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        saved_model_name = "hand_pose_model.pkl"
        feature_importance_graph = "image_string"
        msg = {
            "path": path,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "feature1,feature2",
        }
        event = "train_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = train_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {
            "event": event,
            "status": "success",
            "saved_model_name": saved_model_name,
            "feature_importance_graph": feature_importance_graph,
            "training_accuracy": None,
        }

    def test_train_no_features(self, mocker):
        """
        tests the train_hand_pose function in case of failure (the user has not selected any features)
        should return a status of failed
        """
        path = "path_to_models"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        msg = {
            "path": path,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "",
        }
        event = "train_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = train_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {
            "event": event,
            "status": "failed",
            "error": "Select some features to train the model",
        }

    def test_train_invalid_images(self, mocker):
        """
        tests the train_hand_pose function in case of failure (preprocessing exception)
        should return a status of failed
        """
        mocker.patch.object(HandPoseClassifier, "train")
        mocker.patch.object(HandPoseClassifier, "preprocess").side_effect = Exception
        path = "path_to_model"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        msg = {
            "path": path,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "feature1,feature2",
        }
        event = "train_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = train_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {
            "event": event,
            "status": "failed",
            "error": "Training data contains invalid images",
        }

    def test_train_feature_importace_graph_error(self, mocker):
        """
        tests the train_hand_pose function in case of failure (feature_importance_graph exception)
        should return a status of success
        """
        mocker.patch.object(HandPoseClassifier, "train")
        mocker.patch.object(HandPoseClassifier, "preprocess")
        mocker.patch.object(
            HandPoseClassifier, "feature_importance_graph"
        ).side_effect = Exception
        mocker.patch("handlers.hand_pose_handlers.utils.read_data").return_value = {
            "0": ["image1.png", "image2.png"],
            "1": ["image3.png", "image4.png"],
        }
        mocker.patch.object(HandPoseClassifier, "save")
        path = "path_to_model"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        saved_model_name = "hand_pose_model.pkl"
        msg = {
            "path": path,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "feature1,feature2",
        }
        event = "train_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = train_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {
            "event": event,
            "status": "success",
            "saved_model_name": saved_model_name,
            "feature_importance_graph": "",
            "training_accuracy": None,
        }

    def test_predict_without_training(self, black_image_string):
        """
        tests the predict function without training the model
        should return a prediction of -1
        """
        frame_bytes = black_image_string
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "predict_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = predict_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "prediction": -1}
        print("Predict without training test passed")

    def test_predict_with_training(self, black_image_string, mocker):
        """
        tests the predict function without training the model
        should return a prediction of -1
        """
        mocker.patch.object(HandPoseClassifier, "predict").return_value = 0

        frame_bytes = black_image_string
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "predict_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = predict_hand_pose(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "prediction": 0}
        print("Predict with training test passed")

    def test_preprocess_hand_pose(self, black_image_string):
        """
        tests the preprocess_hand_pose function
        should return a preprocessed image
        """
        frame_bytes = black_image_string
        width = 320
        height = 180
        msg = {"frame": frame_bytes, "width": width, "height": height}
        event = "preprocess_hand_pose"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        result = preprocess_hand_pose(req, res)

        # Assert that the result is as expected
        assert "preprocessed_image" in result
        print("Preprocess test passed")

    def test_load_model_correct_path(self, mocker):
        # mock hand_pose_classifier.load(model_path)
        """
        mock the load function avoid raising an exception for not finding the path
        """
        mocker.patch.object(HandPoseClassifier, "load")
        path = "path_to_model"
        saved_model_name = "hand_pose_model.pkl"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        msg = {
            "path": path,
            "saved_model_name": saved_model_name,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "feature1,feature2",
        }
        event = "load_hand_pose_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = load_hand_pose_model(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "status": "success"}
        print("load model correct path test passed")

    def test_load_model_uncorrect_path(self, mocker):
        # mock hand_pose_classifier.load(model_path)
        """
        mock the load function avoid raising an exception for not finding the path
        """
        path = "path_to_model"
        saved_model_name = "hand_pose_model.pkl"
        model = "SVM"
        feature_extraction_type = "mediapipe"
        msg = {
            "path": path,
            "saved_model_name": saved_model_name,
            "model": model,
            "feature_extraction_type": feature_extraction_type,
            "features": "feature1,feature2",
        }
        event = "load_hand_pose_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        # Call your function
        result = load_hand_pose_model(req, res)

        # Assert that the result is as expected
        assert result == {"event": event, "status": "Model file not found"}
        print("load model uncorrect path test passed")


@pytest.fixture
def black_image_string():
    # read the text from TextImages/black_image.txt
    with open("./Tests/TextImages/black_image.txt") as f:
        return f.read()
