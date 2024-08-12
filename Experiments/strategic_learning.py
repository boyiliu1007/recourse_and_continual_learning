from copy import deepcopy
import numpy as np
from IPython.display import display
from torch import nn, optim

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Models.logisticRegression import LogisticRegression, training
from Models.recourseOriginal import recourse
from Config.config import train, test, sample, model
from Dataset.makeDataset import Dataset

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")

try:
    os.makedirs(DIRECTORY, exist_ok=True)
    print(f"Folder '{DIRECTORY}' is ready.")
except Exception as e:
    print(f"An error occurred: {e}")

class Example10(Helper):
    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)

        size = train.x.shape[0] // 10
        i = np.random.choice(sample.x.shape[0], size, False)
        x = sample.x[i]