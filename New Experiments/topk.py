import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display
import math

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

class Exp2(Helper):
    '''
    1. perform recourse on dataset D
    2. labeling D with topk method
    3. train the model with the updated dataset
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        self.round += 1

        #randomly select from self.sample with size of train and label it with model
        self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'mixed')
        num_false = (isNewList == False).sum().item()
        num_true = (isNewList == True).sum().item()
        print(f"Number of false: {num_false}")
        print(f"Number of true: {num_true}")

        # find training data with label 0 and select 1/5 of them
        data, labels = self.train.x, self.train.y
        label_0_indices = pt.where(labels == 0)[0]
        shuffled_indices = pt.randperm(len(label_0_indices))
        label_0_indices = label_0_indices[shuffled_indices]
        num_samples = len(label_0_indices) // 2
        selected_indices = label_0_indices[:num_samples]

        # perform recourse on the selected subset
        selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
        recourse_weight = getWeights(self.train.x.shape[1], 'uniform')
        print(isNewList[selected_indices].shape)
        recourse(
            self.model,
            selected_subset,
            100,
            recourse_weight,
            loss_list=[],
            threshold=0.7,
            cost_list=self.avgRecourseCost_list,
            q3RecourseCost=self.q3RecourseCost,
            recourseModelLossList=self.recourseModelLossList,
            isNew = isNewList[selected_indices]
        )
        recoursed_data = selected_subset.x
        self.train.x[selected_indices] = recoursed_data
        # use recourse_data, self.train.x[selected_indices]

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
        training(self.model, self.train, 50, self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)

        #calculate metrics
        # TODO: calculate metrics

def getFunc(x):
    # Function definition
    return math.log(x - 0.9) + 2.5
    
def getWeights(feature_num, type = 'uniform'):
    if(type == 'uniform'):
        return pt.from_numpy(np.ones(feature_num)) / feature_num
    elif(type == 'log'):
        weights = np.array([getFunc(i) for i in range(1, feature_num + 1)])
        weights = weights / np.sum(weights)  # Normalize the weights
        return pt.from_numpy(weights)
    else:
        print("type is incorrect at getWeights")

def update_train_data(train, sample, model, type = 'all'):
    if type == 'none':
        return train, pt.empty(0, dtype=pt.bool)

    size = train.x.shape[0]
    sample_indices = pt.randperm(sample.x.shape[0])[:size]
    sampled_x = sample.x[sample_indices]


    if type == 'mixed':
        half_size = size // 2
        retain_indices = pt.randperm(size)[:half_size]
        modify_indices = pt.tensor([i for i in range(size) if i not in retain_indices])

        train.x[modify_indices] = sampled_x[:modify_indices.shape[0]]
        with pt.no_grad():
            y_prob: pt.Tensor = model(train.x[modify_indices])
        y_prob = y_prob.squeeze(1)
        train.y[modify_indices] = pt.where(y_prob > 0.5, 1.0, 0.0)
        isNew = pt.zeros(size, dtype=pt.bool)
        isNew[modify_indices] = True
        
        num_zeros = (train.y == 0).sum().item()
        num_ones = (train.y == 1).sum().item()
        print(f"Number of 0s: {num_zeros}")
        print(f"Number of 1s: {num_ones}")

        return train, isNew


    if type == 'all':
        train.x = sampled_x
        with pt.no_grad():
            y_prob: pt.Tensor = model(train.x)
        y_prob = y_prob.squeeze(1)
        train.y = pt.where(y_prob > 0.5, 1.0, 0.0)

        return train, pt.empty(0, dtype=pt.bool)

    num_zeros = (train.y == 0).sum().item()
    num_ones = (train.y == 1).sum().item()
    print(f"Number of 0s: {num_zeros}")
    print(f"Number of 1s: {num_ones}")

    
exp2 = Exp2(model, pca, train, test, sample)
exp2.save_directory = DIRECTORY
ani1 = exp2.animate_all(80)
ani1.save(os.path.join(DIRECTORY, "ex2.gif"))
exp2.draw_avgRecourseCost()