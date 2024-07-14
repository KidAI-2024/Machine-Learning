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

import matplotlib.pyplot as plt
from pydub import AudioSegment
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  GradientBoostingClassifier
class AudioClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf')
        self.audio_utils = AudioUtils()
        self.class_name_to_index = {}
        self.scaler = StandardScaler()  # Make scaler an instance variable
        self.feature_extraction_methods="chroma_mfccs"
        self.model_name="boost"
        self.kernel_name="rbf"
        self.Knn_neigh="5"
        self.num_est="100"

    def extract_features(self, file_path, audio=False):
        try:
            if audio:
                y = file_path  # If audio is passed directly
                sr = 44100
            else:
                y, sr = librosa.load(file_path, sr=None)

            methods = self.feature_extraction_methods.split('_')
            features = []

            if "mfccs" in methods:
               
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                if mfccs.size > 0:
                    mfccs_mean = np.mean(mfccs, axis=1)
                    features.append(mfccs_mean)
                    
                else:
                    print("MFCCs: Empty feature array detected")

            if "chroma" in methods:
                
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                if chroma.size > 0:
                    chroma_mean = np.mean(chroma, axis=1)
                    features.append(chroma_mean)
                   
                else:
                    print("Chroma: Empty feature array detected")

            if "mel" in methods:
                
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
                if mel_spec.size > 0:
                    mel_spec_mean = np.mean(mel_spec_db, axis=1)
                    features.append(mel_spec_mean)
                   
                else:
                    print("Mel-spectrogram: Empty feature array detected")
            # Ensure only non-empty arrays are concatenated
            if len(features) > 0:
                if isinstance(features[0], np.ndarray):
                    features = [f for f in features if f.size > 0]
                    features = np.concatenate(features)
                
                return features
            else:
                raise ValueError("All extracted features are empty.")

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
        output_directory=dir
        output_directory = self.audio_utils.preprocess_all(train_directory)
        self.audio_utils.process_main_directory(output_directory)
        X_train, y_train, class_labels_train = self.load_dataset(output_directory)

        if X_train.size == 0:
            raise ValueError("No features were extracted, please check your dataset and feature extraction.")

        # X_train=  X_train/ 32768.0
        self.class_name_to_index = {name: i for i, name in enumerate(sorted(set(class_labels_train)))}

        y_train_mapped = np.array([self.class_name_to_index[class_labels_train[i]] for i in y_train])

        # Preprocess data (Normalization)
        X_train = self.scaler.fit_transform(X_train)

        joblib.dump(self.scaler, 'scaler.pkl')
        
        if self.model_name=="svm":
            print("svmmmmmmmmm")
            print("kernel ",self.kernel_name)
            self.model=SVC(kernel=self.kernel_name)

        if self.model_name=="knn":
             self.model = KNeighborsClassifier(n_neighbors=int(self.Knn_neigh))
        
        if self.model_name=="boost":
             self.model = GradientBoostingClassifier(n_estimators=int(self.num_est))

        # Train a classifier
        self.model.fit(X_train, y_train_mapped)

        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train_mapped, train_predictions)
        print(f"Training accuracy : {int(train_accuracy*100)}")
        # Print confusion matrix
        conf_matrix = confusion_matrix(y_train_mapped, train_predictions)
        print("Confusion Matrix :")
        print(conf_matrix)
        # self.pred_test(dir)
        # self.pred_test(dir)

        
        print("Training completed successfully.")
        return int(train_accuracy*100)


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
                   
                        predicted_class_index = self.predict(file_path)
                        
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
        # t1=time.time()
        wav_audio = self.audio_utils.convert_to_wav(audio)
        # t2=time.time()
        print("after wac")
        prep_audio = self.audio_utils.preprocess_audio(audio)#here problem
        print("after prepross")
        # t3=time.time()
        features = self.extract_features(audio)
        print("after extract")
        # t4=time.time()
        # print("first time ",t2-t1)
        # print("second time ",t3-t2)
        # print("first time ",t4-t3)
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
        with open(path, "wb") as model_file:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_extraction_methods": self.feature_extraction_methods,
                "model_name": self.model_name,
                "kernel_name": self.kernel_name,
                "knn_neigh": self.Knn_neigh,
                "num_est": self.num_est
            }, model_file)

    def load(self, path):
        with open(path, "rb") as model_file:
            
            data = pickle.load(model_file)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_extraction_methods = data["feature_extraction_methods"]
            self.model_name = data["model_name"]
            self.kernel_name = data["kernel_name"]
            self.Knn_neigh = data["knn_neigh"]
            self.num_est = data["num_est"]
            print(data)
        print("Model and scaler loaded successfully from", path)
