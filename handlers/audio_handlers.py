from AudioClassifier.audio_classifier import AudioClassifier
import utils
from decorators import event
from server_utils import Req, Res
import os
from io import BytesIO
import librosa
import base64
import IPython.display as ipd
import soundfile as sf
from pydub import AudioSegment

audio_classifier = AudioClassifier()


@event("start_audio_train")
def train_audio(req: Req, res: Res) -> int:
    path = req.msg["path"]
    path = os.path.abspath(os.path.join(path, '../../../Engine', path))
        
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    print("training data...")
    
# try:
    audio_classifier.train(path)
# except Exception as e:
#     print(f"Error in training: {e}")
    # res_msg = {"status": "failed"}
    # return res.build(req.event, res_msg)
  
    print("Saving model...")
    project_name = path.split("/")[-1]
    saved_model_name = "audio_model.pkl"
    model_path = os.path.join(
        "..", "Engine", "Projects", project_name, saved_model_name
    )  # Currect directory is Machine-Learning
    audio_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {"status": "success", "saved_model_name": saved_model_name}
    return res.build(req.event, res_msg)




@event("predict_audio")
def predict_audio(req: Req, res: Res):
    print("IN predict audio")
   
    audio_base64 = req.msg["audio"]
    sample_rate = int(req.msg["sample_rate"])
    audio_bytes = base64.b64decode(audio_base64)

    # Save the received audio data (y) to a file
    with open("received_audio.wav", "wb") as f:
        f.write(audio_bytes)

    # Load the audio from the saved file
    y, sr = librosa.load("received_audio.wav", sr=sample_rate)

    # Optionally, you can save the loaded audio to another file for verification
    sf.write("loaded_audio.wav", y, sr)
    
    audio_segment = AudioSegment(
        y.tobytes(), 
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1  # librosa loads audio in mono
    )
        
    
    pred = audio_classifier.predict(audio_segment)
    # except Exception as e:
    #     print(f"Error in predict: {e}")
    #     pred = -1
    print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)


@event("load_audio_model")
def load_audio_model(req: Req, res: Res) -> int:
    print("in loading ..........")
    project_name = req.msg["project_name"]
    saved_model_name = req.msg["saved_model_name"]
    if not saved_model_name:
       saved_model_name= "audio_model.pkl"
    print("saveddddddddd modelllllllll nameddddd ",saved_model_name)
    print("saveddddddddd projecttttt nameddddd ",project_name)
    model_path = os.path.join(
        "..", "Engine", "Projects", project_name, saved_model_name
    )
    try:
        
        audio_classifier.load(model_path)
        print(f"Model loaded from {model_path}")
        res_msg = {"status": "success"}
    except Exception as e:
        res_msg = {"status": "Model file not found"}

    return res.build(req.event, res_msg)
