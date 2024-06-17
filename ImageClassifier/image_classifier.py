from skimage.feature import hog
import numpy as np

from sklearn.svm import SVC
import pickle

class ImageClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear")

    def preprocess(self, images):
        pass

    def train(self, data):
        pass

    def predict(self, data):
        pass
    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        self.model = pickle.load(open(path, "rb"))
