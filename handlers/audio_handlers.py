from AudioClassifier.audio_classifier import AudioClassifier
from decorators import event
from server_utils import Req, Res
import os
import librosa
import base64
import soundfile as sf
from pydub import AudioSegment
import pyaudio
import numpy as np
import threading
import shutil
import webrtcvad
import resampy
import time
from datetime import datetime
import wave
audio_classifier = AudioClassifier()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define audio parameters
sample_rate = 32000  # Desired sample rate for audio processing and saving
chunk_duration = 0.03 # Duration of each chunk in seconds (20ms)
chunk_size = int(sample_rate * chunk_duration)
# audio_save_dir = "audios"

# Global variables for mic stream and prediction flag
stream = None
stop_prediction = False
prediction_started = False
audio_thread = None
ind = 0

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0: least aggressive, 3: most aggressive

@event("start_audio_train")
def train_audio(req: Req, res: Res) -> int:
    path = req.msg["path"]
    
    # Training data is map {"Class Number(first character in the folder name)" : [images]}
    print("Training data...")

    audio_classifier.train(path)

    print("Saving model...")
    project_name = path.split("/")[-1]
    saved_model_name = "audio_model.pkl"
    model_path = os.path.join(path, project_name, saved_model_name)   # Current directory is Machine-Learning
    audio_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {"status": "success", "saved_model_name": saved_model_name, "dum": "dummy"}
    return res.build(req.event, res_msg)


@event("load_audio_model")
def load_audio_model(req: Req, res: Res) -> int:
    print("In loading...")
    # y, sr = librosa.load("accumulated_audio.wav")
    # dummy_audio = np.zeros(44100, dtype=np.float32)  # 1 second of silence at 44100 Hz

    # # Perform a dummy resampling to warm up the process
    # resampy.resample(dummy_audio, 44100, 32000)
    y, sr = librosa.load("accumulated_audio.wav")
    dummy_audio = np.zeros(44100, dtype=np.float32)  # 1 second of silence at 44100 Hz

    # Perform a dummy resampling to warm up the process
    resampy.resample(dummy_audio, 44100, 32000) 
   
    path = req.msg["path"]
    saved_model_name = req.msg["saved_model_name"]
    if not saved_model_name:
        saved_model_name = "audio_model.pkl"
    model_path = os.path.join(path, saved_model_name)
    try:
        audio_classifier.load(model_path)
        print(f"Model loaded from {model_path}")
        res_msg = {"status": "success"}
    except Exception as e:
        res_msg = {"status": "Model file not found"}
    
    return res.build(req.event, res_msg)

def start_audio_capture_thread(req: Req, res: Res):
    global stop_prediction, ind
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    print("Streaming audio...")

    audio_data_accumulated = b""
    speaking = False

    while not stop_prediction:
      
        audio_data = stream.read(chunk_size)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        frames = [audio_np[i:i + chunk_size] for i in range(0, len(audio_np), chunk_size)]
        speech_detected = False

        for frame in frames:
            if len(frame) == chunk_size and vad.is_speech(frame.tobytes(), sample_rate):
                speech_detected = True
                break
       
        if speech_detected:
            print("Voice detected")
            speaking = True
            audio_data_accumulated += audio_data

        else:
            if speaking:
               
                speaking = False
                if audio_data_accumulated:
                    # Resample the accumulated audio data to 44100 Hz before saving
                    audio_np_accumulated = np.frombuffer(audio_data_accumulated, dtype=np.int16).astype(np.float32) 
                    print("before resamp1")
                    #here prob
                    audio_np_resampled = resampy.resample(audio_np_accumulated, sample_rate, 44100)
                    print("before resamp2")
                    # Ensure the audio is exactly 1 second long
                    if len(audio_np_resampled) > 44100:
                        audio_np_resampled = audio_np_resampled[:44100]
                    elif len(audio_np_resampled) < 44100:
                        padding = 44100 - len(audio_np_resampled)
                        audio_np_resampled = np.pad(audio_np_resampled, (0, padding), 'constant')

                    audio_np_resampled = audio_np_resampled.astype(np.int16)
                    sf.write("accumulated_audio.wav", audio_np_resampled, 44100)
                    
                    # y, sr = librosa.load("accumulated_audio.wav", sr=44100)
                    print("before pred")
                    pred = audio_classifier.predict("accumulated_audio.wav")
                    
                    # filename = f"{pred}_loaded_audio_{ind}.wav"
                    # file_path = os.path.join(audio_save_dir, filename)
                    # sf.write(file_path, y, sr)
                    ind += 1

                    print(f"Predicted class: {pred}")
                    res_msg = {"prediction": str(pred)}
                    audio_data_accumulated = b""  # Reset accumulated audio
                    current_time = datetime.now()

                    # Print the high-accuracy current time
                    print(f"sending pred time: {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    res.send(req.event, res_msg)

    stream.stop_stream()
    stream.close()


# Function to handle start prediction event
@event("predict_audio")
def predict_audio(req: Req, res: Res):
    global prediction_started, audio_thread, stop_prediction

    # Clear the directory if it exists, otherwise create it
    

    if not prediction_started:
        stop_prediction = False
        prediction_started = True
        # Start audio capture in a separate thread
        audio_thread = threading.Thread(target=start_audio_capture_thread, args=(req, res))
        audio_thread.start()


# Function to handle stop prediction event
@event("stop_prediction")
def stop_audio_capture(req: Req, res: Res):
    print("Stop Predictionnnnn")
    global stop_prediction, prediction_started, audio_thread
    stop_prediction = True
    prediction_started = False
    if audio_thread:
        audio_thread.join()  # Ensure thread finishes before returning
