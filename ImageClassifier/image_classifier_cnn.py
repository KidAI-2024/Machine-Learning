import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import pickle
import os


class ImageClassifierCNN:
    def __init__(self):
        pass

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
