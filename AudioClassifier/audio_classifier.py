import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from .audio_utils import AudioUtils
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pydub import AudioSegment



class AudioClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear')
        self.audio_utils = AudioUtils()
        self.class_name_to_index = {}
        self.scaler = StandardScaler()  # Make scaler an instance variable

    def extract_features(self, file_path, audio=False):
        try:
            if audio:
                mfccs = librosa.feature.mfcc(y=file_path, sr=44100, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                return mfccs_mean
            else:
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                return mfccs_mean
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None

    def load_dataset(self, directory):
        features = []
        labels = []
        file_paths = []
        class_indices = {}
        class_labels = []

        for class_index, class_name in enumerate(os.listdir(directory)):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):  # Check if it's a directory
                if class_name not in class_indices:
                    class_indices[class_name] = len(class_indices)
                    class_labels.append(class_name)
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if file_path.endswith('.wav'):
                        feature_vector = self.extract_features(file_path)
                        if feature_vector is not None:
                            features.append(feature_vector)
                            labels.append(class_indices[class_name])
                            file_paths.append(file_path)
            else:
                print(f"Skipping non-directory file: {class_path}")

        return np.array(features), np.array(labels), class_labels

    def train(self, dir):
        train_directory = dir  # Replace with the path to your training directory
        print(dir)
        items = os.listdir(train_directory)
        
        # Filter out the items that are directories
        folders = [item for item in items if os.path.isdir(os.path.join(train_directory, item))]
        
        # Print the names of the folders
        for folder in folders:
            print(folder)
            
        self.audio_utils.convert_all_to_wav(train_directory)
        output_directory = self.audio_utils.preprocess_all(train_directory)
        # self.audio_utils.process_main_directory(output_directory)
        X_train, y_train, class_labels_train = self.load_dataset(output_directory)

        if X_train.size == 0:
            raise ValueError("No features were extracted, please check your dataset and feature extraction.")

        # X_train=  X_train/ 32768.0
        self.class_name_to_index = {name: i for i, name in enumerate(sorted(set(class_labels_train)))}

        y_train_mapped = np.array([self.class_name_to_index[class_labels_train[i]] for i in y_train])

        # Preprocess data (Normalization)
        X_train = self.scaler.fit_transform(X_train)

        joblib.dump(self.scaler, 'scaler.pkl')

        # Train a classifier
        self.model.fit(X_train, y_train_mapped)

        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train_mapped, train_predictions)
        print(f"Training accuracy : {train_accuracy:.4f}")
        # Print confusion matrix
        conf_matrix = confusion_matrix(y_train_mapped, train_predictions)
        print("Confusion Matrix :")
        print(conf_matrix)
        self.pred_test(dir)

        
        print("Training completed successfully.")


    def pred_test(self, directory):
        items = os.listdir(directory)
        
        # Filter out the items that are directories
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        
        # Initialize lists to store predictions and true labels
        predictions = []
        true_labels = []

        # Iterate over each class directory
        for folder in folders:
            class_path = os.path.join(directory, folder)
            
            # Iterate over each file in the class directory
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                
                # Check if the file is a .wav file
                if file_path.endswith('.wav'):
                    # try:
                        # Load the audio file with librosa
                        audio_data, sr = librosa.load(file_path, sr=None)
                        
                        # Convert audio data to AudioSegment
                        # audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=sr)
                        audio_segment = AudioSegment(
                            audio_data.tobytes(), 
                            frame_rate=sr,
                            sample_width=audio_data.dtype.itemsize,
                            channels=1  # librosa loads audio in mono
                        )
                        # Predict class using your predict method (adjust as per your actual method)
                        predicted_class_index = self.predict(audio_segment)
                        
                        # Store predictions and true labels
                        predictions.append(predicted_class_index)
                        true_labels.append(self.class_name_to_index[folder])
                        
                    # except Exception as e:
                    #     print(f"Error processing {file_path}: {e}")
    
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Total accuracy on training data: {accuracy:.4f}")
        conf_matrix = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix :")
        print(conf_matrix)
        return accuracy



    def predict(self, audio):
        # print("class map",self.class_name_to_index)
        wav_audio = self.audio_utils.convert_to_wav(audio)
        prep_audio = self.audio_utils.preprocess_audio(wav_audio)
        features = self.extract_features(prep_audio, True)
        
        if features is None or features.size == 0:
            raise ValueError("Feature extraction failed. No features to predict.")
        
        # print("features beofre reashping",features)
        features = features.reshape(1, -1)  # Ensure the features have the correct shape for prediction
        # print("features after reashping",features)
        features = self.scaler.transform(features)  # Scale the features

        pred = self.model.predict(features)
        class_index = pred[0]
        # print(pred)
        return class_index

    def save(self, path):
        """Save the model and scaler to disk"""
        with open(path, "wb") as model_file:
            pickle.dump({"model": self.model, "scaler": self.scaler}, model_file)

    def load(self, path):
        """Load the model and scaler from disk"""
        # print("inseifsdfadfadfadfadfa",path)
        with open(path, "rb") as model_file:
            print("path opened ",path)
            data = pickle.load(model_file)
            self.model = data["model"]
            self.scaler = data["scaler"]
            print("in loading ..........",self.scaler)
# Assuming AudioUtils class is defined elsewhere with necessary methods.
