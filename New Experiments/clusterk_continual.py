import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display
from scipy.stats import gaussian_kde
import math

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca

from Models.logisticRegression import LogisticRegression, training
from Models.synapticIntelligence import continual_training
from Models.recourseGradient import recourse
from Config.continual_config import train, test, sample, si, POSITIVE_RATIO
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

class Exp3(Helper):
    '''
    1. perform recourse on dataset D
    2. labeling D with diversek method
    3. continual training the model with the updated dataset
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        self.round += 1

        #randomly select from self.sample with size of train and label it with model
        size = self.train.x.shape[0]
        sample_indices = pt.randperm(self.sample.x.shape[0])[:size]
        self.train.x = self.sample.x[sample_indices]
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(self.train.x)
        y_prob = y_prob.squeeze(1)
        self.train.y = pt.where(y_prob > 0.5, 1.0, 0.0)
        num_zeros = (self.train.y == 0).sum().item()
        num_ones = (self.train.y == 1).sum().item()
        print(f"Number of 0s: {num_zeros}")
        print(f"Number of 1s: {num_ones}")

        # find training data with label 0 and select 1/5 of them
        data, labels = self.train.x, self.train.y
        label_0_indices = pt.where(labels == 0)[0]
        shuffled_indices = pt.randperm(len(label_0_indices))
        label_0_indices = label_0_indices[shuffled_indices]
        num_samples = len(label_0_indices) // 5
        selected_indices = label_0_indices[:num_samples]

        # perform recourse on the selected subset
        selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
        # recourse_weight = pt.from_numpy(np.ones(self.train.x.shape[1])) / self.train.x.shape[1]
        recourse_weight = pt.from_numpy(getWeights(self.train.x.shape[1]))
        recourse(self.model, selected_subset, 100,recourse_weight,loss_list=[],threshold=0.5,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
        recoursed_data = selected_subset.x
        self.train.x[selected_indices] = recoursed_data

        # update the labels of D using diversek method
        with pt.no_grad():
          y_prob_all: pt.Tensor = self.model(self.train.x)
        positive_indices = pt.where(y_prob_all > 0.5)[0]
        positive_data = data[positive_indices]
        kde = gaussian_kde(positive_data.cpu().numpy().T)  # Transpose for KDE input
        kde_scores = kde(positive_data.cpu().numpy().T)    # Compute density
        weights = kde_scores / np.sum(kde_scores)
        sampled_indices = np.random.choice(
            positive_indices.cpu().numpy(),  # Indices to sample from
            size=self.train.x.shape[0]//2,                          # Number of samples
            replace=False,                   # No replacement
            p=weights                        # Probability weights
        )
        self.train.y = pt.zeros(self.train.y.shape)
        self.train.y[sampled_indices] = 1

        
        # remove the unselected positive data from the training set
        unselected_indices = np.setdiff1d(positive_indices.cpu().numpy(), sampled_indices)
        # Store unselected positive data
        unselected_x = self.train.x[unselected_indices]  
        unselected_y = self.train.y[unselected_indices]
        # Remove only unselected positive data from training set
        keep_indices = np.setdiff1d(np.arange(self.train.x.shape[0]), unselected_indices)
        self.train.x = self.train.x[keep_indices]             
        self.train.y = self.train.y[keep_indices]             

        # train the model with the updated dataset
        if(self.jsd_list == []):
            continual_training(self.si, self.train, 50, lambda_ = 0)
        else:
            continual_training(self.si, self.train, 50, lambda_ = 10/(self.jsd_list[-1]))

        #add back the unselected positive data
        self.train.x = pt.cat((self.train.x, unselected_x))
        self.train.y = pt.cat((self.train.y, unselected_y))

        #calculate metrics


def getFunc(x):
    # Function definition
    return math.log(x - 0.9) + 2.5
    
def getWeights(feature_num):
    weights = np.array([getFunc(i) for i in range(1, feature_num + 1)])
    weights = weights / np.sum(weights)  # Normalize the weights
    return weights

exp3 = Exp3(si.model, pca, train, test, sample)
exp3.si = si
exp3.save_directory = DIRECTORY
ani1 = exp3.animate_all(80)
ani1.save(os.path.join(DIRECTORY, "ex3.gif"))