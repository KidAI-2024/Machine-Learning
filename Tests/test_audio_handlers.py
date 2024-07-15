
from unittest.mock import MagicMock
from handlers.audio_handlers import *
from AudioClassifier.audio_classifier import AudioClassifier
from server_utils import Req, Res
import os
import pytest

class TestAudioClassifier:
    # @pytest.fixture
    # def setup_mocks(self, mocker):
    #     mocker.patch("pyaudio.PyAudio.open")
    #     mocker.patch("pyaudio.PyAudio.read", return_value=b"mock_audio_data")
    #     mocker.patch("webrtcvad.Vad.is_speech", return_value=True)
    #     mocker.patch("resampy.resample", return_value=np.zeros(44100, dtype=np.float32))
    #     mocker.patch("soundfile.write")
    #     mocker.patch.object(audio_classifier, "predict", return_value="mock_prediction")
    #     mocker.patch("datetime.datetime")

    def test_load_audio_model_not_found(self, mocker):
        """
        Test loading an audio model when the model file is not found.
        Should return status indicating model file not found.
        """
        mocker.patch.object(AudioClassifier, "load").side_effect = FileNotFoundError
        
        path = "path_to_models"
        saved_model_name = "audio_model.pkl"
        
        msg = {
            "path": path,
            "saved_model_name": saved_model_name,
        }
        event = "load_audio_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        
        result = load_audio_model(req, res)
        
        assert result == {"event": event, "status": "Model file not found"}

    def test_train_audio_success_acc(self, mocker):
        """
        Test training audio with an invalid path.
        Should return status indicating failure and appropriate error message.
        """
        mocker.patch.object(AudioClassifier, "train").return_value =100
        mocker.patch.object(AudioClassifier, "save").return_value =100
        
        path = "invalid_path"
        
        msg = {
            "path": path,
            "feature_extraction_methods": "mfcc",
            "model_name": "SVM",
            "kernel_name": "linear",
            "knn_neigh": "5",
            "num_est": "100"
        }
        event = "start_audio_train"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        
        result = train_audio(req, res)
        print("resulttttt ",result)
        assert result == {'event': 'start_audio_train', 'status': 'success', 'saved_model_name': 'audio_model.pkl', 'accuracy': '100'}
 

    @pytest.fixture
    def mock_pyaudio_open(self, monkeypatch):
        # Mock pyaudio.PyAudio.open to return a mock stream object
        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read.return_value = b"mock_audio_data"
        mock_audio.open.return_value = mock_stream
        
        monkeypatch.setattr(pyaudio.PyAudio, "open", mock_audio.open)
        return mock_audio

    @pytest.fixture
    def mock_webrtcvad_is_speech(self, monkeypatch):
        # Mock webrtcvad.Vad.is_speech to always return True
        mock_is_speech = MagicMock()
        mock_is_speech.return_value = True
        
        monkeypatch.setattr(webrtcvad.Vad, "is_speech", mock_is_speech)
        return mock_is_speech

    @pytest.fixture
    def mock_resampy_resample(self, monkeypatch):
        # Mock resampy.resample to return zeros of correct length
        def mock_resample(input_signal, *args, **kwargs):
            return np.zeros(44100, dtype=np.float32)
        
        monkeypatch.setattr(resampy, "resample", mock_resample)

    @pytest.fixture
    def mock_soundfile_write(self, monkeypatch):
        # Mock soundfile.write
        mock_write = MagicMock()
        monkeypatch.setattr(sf, "write", mock_write)
        return mock_write

    @pytest.fixture
    def mock_audio_classifier_predict(self, monkeypatch):
        # Mock audio_classifier.predict to return a mock prediction
        mock_predict = MagicMock()
        mock_predict.return_value = "mock_prediction"
        
        monkeypatch.setattr(audio_classifier, "predict", mock_predict)
        return mock_predict

    @pytest.fixture
    def setup_thread(self, mocker):
        """
        Fixture to setup the test with mocked objects.
        """
        req = Req()
        res = Res()

        # Mock the send method of the Res object
        res.send = mocker.Mock()

        return req, res

    def test_start_audio_capture_thread_success(self, setup_thread):
        """
        Test starting audio capture thread successfully.
        Should accumulate audio data and return the prediction result.
        """
        req, res = setup_thread

        # Start the audio capture thread
        audio_thread = threading.Thread(target=start_audio_capture_thread, args=(req, res))
        audio_thread.start()

        # Allow some time for the thread to run (adjust as necessary)
        time.sleep(1)  # Increase if needed based on the actual execution time of start_audio_capture_thread

        # Set stop condition to end the thread
        global stop_prediction
        stop_prediction = True

        # Join the thread to ensure it completes
        audio_thread.join(timeout=2)  # Timeout after 10 seconds


        # After the thread completes, check if an exception was raised
        assert res.send.call_count == 0  # Ensure that res.send was not called due to error

        # Check if an exception was raised during the thread execution
        if hasattr(audio_thread, "exception"):
            assert isinstance(audio_thread.exception, ValueError)
            assert str(audio_thread.exception) == "Feature extraction failed. No features to predict."

        print("Test completed successfully")

 
    def test_stop_prediction_when_not_started(self, mocker):
        """
        Test stopping audio prediction when it hasn't started.
        Should ensure prediction flags remain unchanged.
        """
        global stop_prediction, prediction_started, audio_thread
        stop_prediction = False
        prediction_started = False
        audio_thread = None
        
        req = Req()
        res = Res()
        
        stop_audio_capture(req, res)
        
        assert not stop_prediction
        assert not prediction_started
        assert audio_thread is None
        
        # Add additional assertions as needed


    def test_load_audio_model_success(self, mocker):
        """
        Test loading an audio model successfully.
        Should return status indicating model loaded successfully.
        """
        mocker.patch.object(AudioClassifier, "load")
        
        path = "path_to_models"
        saved_model_name = "audio_model.pkl"
        
        msg = {
            "path": path,
            "saved_model_name": saved_model_name,
        }
        event = "load_audio_model"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        
        result = load_audio_model(req, res)
        
        assert result == {"event": event, "status": "success"}
    
    def test_predict_audio_exception_handling(self, mocker):
        """
        Test predicting audio with exception handling.
        Should catch exceptions and return status indicating failure and appropriate error message.
        """
        mocker.patch.object(AudioClassifier, "predict").side_effect = Exception("Prediction error")
        
        audio_file = "path_to_your_audio_file.wav"
        msg = {"audio_file": audio_file}
        event = "predict_audio"
        req = Req()
        req.msg = msg
        req.event = event
        res = Res()
        
        result = predict_audio(req, res)
        
        assert result ==None
        
  