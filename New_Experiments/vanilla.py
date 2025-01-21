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
from Models.recourseGradient import recourse
from Config.config import train, test, sample, model, POSITIVE_RATIO
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

class Exp1(Helper):
    '''
    1. perform recourse on dataset D
    2. train the model with the updated dataset
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
        num_samples = len(label_0_indices) // 2
        selected_indices = label_0_indices[:num_samples]

        # perform recourse on the selected subset
        selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
        if len(selected_subset.x) == 0:
            print("No data selected for recourse")
            return
        else:
            recourse_weight = pt.from_numpy(np.ones(self.train.x.shape[1])) / self.train.x.shape[1]
            recourse(self.model, selected_subset, 100,recourse_weight,loss_list=[],threshold=0.5,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
            recoursed_data = selected_subset.x
            self.train.x[selected_indices] = recoursed_data

            #if data in train.y[selected_indices] > 0.5, set it to 1, else set it to 0
            with pt.no_grad():
                y_prob_recourse: pt.Tensor = self.model(self.train.x[selected_indices])
            y_prob_recourse = y_prob_recourse.squeeze(1)
            self.train.y[selected_indices] = pt.where(y_prob_recourse > 0.5, 1.0, 0.0)


        # train the model with the updated dataset
        training(self.model, self.train, 50,self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)

        #calculate metrics
        # TODO: calculate metrics (fail to recourse, st-acc, st-fwt, st-bwt, jsd)


exp1 = Exp1(model, pca, train, test, sample)
exp1.save_directory = DIRECTORY
ani1 = exp1.animate_all(480)
ani1.save(os.path.join(DIRECTORY, "ex1.gif"))
