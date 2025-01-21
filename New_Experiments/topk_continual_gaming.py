import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display

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
    2. labeling D with topk method
    3. continual training the model with the updated dataset
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        self.round += 1

        # find training data with label 0 and select 1/5 of them
        data, labels = self.train.x, self.train.y
        label_0_indices = pt.where(labels == 0)[0]
        shuffled_indices = pt.randperm(len(label_0_indices))
        label_0_indices = label_0_indices[shuffled_indices]
        num_samples = len(label_0_indices) // 5
        selected_indices = label_0_indices[:num_samples]

        # perform recourse on the selected negative subset
        selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
        recourse_weight = pt.from_numpy(np.ones(self.train.x.shape[1])) / self.train.x.shape[1]
        recourse(self.model, selected_subset, 100,recourse_weight,loss_list=[],threshold=0.5,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
        recoursed_data = selected_subset.x
        self.train.x[selected_indices] = recoursed_data

        #perform recourse on the selected positive subset
        label_1_indices = pt.where(labels == 1)[0]
        with pt.no_grad():
            y_prob_1: pt.Tensor = self.model(data[label_1_indices])
        recourse_indices = pt.where(y_prob_1 < 0.9)[0]
        
        selected_subset = Dataset(data[recourse_indices], labels[recourse_indices].unsqueeze(1))
        recourse(self.model, selected_subset, 100,recourse_weight,loss_list=[],threshold=0.9,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
        recoursed_data = selected_subset.x
        self.train.x[recourse_indices] = recoursed_data

        # update the labels of D using topk method
        with pt.no_grad():
          y_prob_all: pt.Tensor = self.model(self.train.x)
        y_prob_all = y_prob_all
        sorted_indices = pt.argsort(y_prob_all[:, 0], dim=0, descending=True)
        cutoff_index = int(len(sorted_indices) * POSITIVE_RATIO)
        mask = pt.zeros_like(y_prob_all)
        mask[sorted_indices[:cutoff_index]] = 1
        self.train.y = mask.float().squeeze(1)

        # train the model with the updated dataset
        if(self.jsd_list == []):
            continual_training(self.si, self.train, 50, lambda_ = 0)
        else:
            continual_training(self.si, self.train, 50, lambda_ = 0.1/(self.jsd_list[-1]))

        #calculate metrics
        # TODO: calculate metrics


exp3 = Exp3(si.model, pca, train, test, sample)
exp3.si = si
exp3.save_directory = DIRECTORY
ani1 = exp3.animate_all(80)
ani1.save(os.path.join(DIRECTORY, "ex3.gif"))
