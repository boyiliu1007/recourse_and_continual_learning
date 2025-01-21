import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display
from scipy.stats import gaussian_kde
import math
import datetime

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Experiment_Helper.auxiliary import getWeights, update_train_data, FileSaver

from Models.logisticRegression import LogisticRegression, training
from Models.synapticIntelligence import continual_training
from Models.recourseGradient import recourse
from Config.continual_config import train, test, sample, si, dataset, POSITIVE_RATIO # modified parameters for observations
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

# modified parameters for observations
THRESHOLD = 0.5
RECOURSENUM = 0.5
COSTWEIGHT = 'uniform'
DATASET = dataset

class Exp3(Helper):
    '''
    1. perform recourse on dataset D
    2. labeling D with preservedk method
    3. continual training the model with the updated dataset
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        self.round += 1

        #save model parameters
        self.model_params = deepcopy(self.model.state_dict())

        #randomly select from self.sample with size of train and label it with model
        self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'mixed', self.train_size)

        # find training data with label 0 and select 1/5 of them
        data, labels = self.train.x, self.train.y
        label_0_indices = pt.where(labels == 0)[0]
        shuffled_indices = pt.randperm(len(label_0_indices))
        label_0_indices = label_0_indices[shuffled_indices]
        num_samples = math.floor(len(label_0_indices) * RECOURSENUM)
        selected_indices = label_0_indices[:num_samples]

        # perform recourse on the selected subset
        selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
        recourse_weight = getWeights(self.train.x.shape[1], COSTWEIGHT)
        
        recourse(
            self.model,
            selected_subset,
            100,
            recourse_weight,
            loss_list=[],
            threshold=THRESHOLD,
            cost_list=self.avgRecourseCost_list,
            q3RecourseCost=self.q3RecourseCost,
            recourseModelLossList=self.recourseModelLossList,
            isNew = isNewList[selected_indices],
            new_cost_list=self.avgNewRecourseCostList,
            original_cost_list=self.avgOriginalRecourseCostList
        )
        
        recoursed_data = selected_subset.x
        self.train.x[selected_indices] = recoursed_data

        # update the labels of D using preservedk method
        with pt.no_grad():
          y_prob_all: pt.Tensor = self.model(self.train.x)
        positive_indices = pt.where(y_prob_all > 0.5)[0]
        positive_data = data[positive_indices]
        kde = gaussian_kde(positive_data.cpu().numpy().T)  # Transpose for KDE input
        kde_scores = kde(positive_data.cpu().numpy().T)
        kde_scores = kde_scores + y_prob_all[positive_indices].cpu().numpy().flatten()
        weights = kde_scores / np.sum(kde_scores)

        sampled_indices = np.random.choice(
            positive_indices.cpu().numpy(),  # Indices to sample from
            size=math.floor(self.train.x.shape[0] * POSITIVE_RATIO),                         # Number of samples
            replace=False,                   # No replacement
            p=weights                        # Probability weights
        )
        self.train.y = pt.zeros(self.train.y.shape)
        self.train.y[sampled_indices] = 1

        # Find additional indices where 0.5 < y_prob_all ≤ 0.6
        extra_indices = pt.where((y_prob_all > 0.5) & (y_prob_all <= 0.7))[0]
        sampled_indices = pt.tensor(sampled_indices, dtype=pt.long)
        sampled_indices = pt.unique(pt.cat((sampled_indices, extra_indices)))
        sampled_indices = sampled_indices.numpy()

        # remember the size of the training set
        self.train_size = self.train.x.shape[0]
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
            continual_training(self.si, self.train, 50, lambda_ = 0.0000001/(self.jsd_list[-1]))

        self.si.update_omega(self.train, nn.BCELoss())
        self.si.consolidate(3)

        #calculate metrics: ========================================================================
        #calculate short term accuracy
        current_data = Dataset(self.train.x, self.train.y)
        self.historyTrainList.append(current_data)
        self.overall_acc_list.append(self.calculate_AA(self.model, self.historyTrainList, 4))
        
        #add back the unselected positive data in original order
        # Create a new tensor with the correct shape
        new_x = pt.zeros((self.train.x.shape[0] + unselected_x.shape[0], *self.train.x.shape[1:]), dtype=self.train.x.dtype)
        new_y = pt.zeros((self.train.y.shape[0] + unselected_y.shape[0], *self.train.y.shape[1:]), dtype=self.train.y.dtype)
        # Fill in the values
        new_x[keep_indices] = self.train.x  # Place the kept data
        new_y[keep_indices] = self.train.y
        new_x[unselected_indices] = unselected_x  # Insert unselected data back in the correct spots
        new_y[unselected_indices] = unselected_y
        self.train.x = new_x
        self.train.y = new_y

        #calculate ftr
        recourseFailCnt = pt.where(self.train.y[selected_indices] == 0)[0].shape[0]
        recourseFailRate = recourseFailCnt / len(self.train.y[selected_indices])
        self.failToRecourse.append(recourseFailRate)

        #jsd is calculated in helper.py already

        #calculate t_rate
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(test.x)
        #calculate the ratio of 1s and 0s in the test data
        num_ones = pt.where(y_prob > 0.5)[0].shape[0]
        num_zeros = len(y_prob) - num_ones
        t_rate = num_ones / num_zeros
        self.t_rate_list.append(t_rate)
        print("t_rate: ",t_rate)

        #calculate model shift distance
        last_model_params = self.model_params
        current_model_params = self.model.state_dict()
        shift_distance = pt.norm(
            pt.cat([pt.flatten(last_model_params[key] - current_model_params[key])
                for key in last_model_params.keys()]), p=2
        )
        self.model_shift_distance_list.append(shift_distance)
        #===========================================================================================
        
        self.train.x = self.train.x[keep_indices]             
        self.train.y = self.train.y[keep_indices]




exp3 = Exp3(si.model, pca, train, test, sample)
exp3.si = si
exp3.save_directory = DIRECTORY
current_time = datetime.datetime.now().strftime("%d-%H-%M")
ani1 = exp3.animate_all(100)
ani1.save(os.path.join(DIRECTORY, f"{RECOURSENUM}_{THRESHOLD}_{POSITIVE_RATIO}_{COSTWEIGHT}_{DATASET}_{current_time}.mp4"))
exp3.overall_acc_list = exp3.overall_acc_list[3:]
exp3.draw_avgRecourseCost()
exp3.plot_jsd()
exp3.draw_Fail_to_Recourse()
exp3.plot_aac()
exp3.plot_t_rate()
exp3.plot_model_shift()

# save to csv
FileSaver(exp3.failToRecourse, 
          exp3.overall_acc_list, 
          exp3.jsd_list, 
          exp3.avgRecourseCost_list, 
          exp3.avgNewRecourseCostList, 
          exp3.avgOriginalRecourseCostList,
          exp3.t_rate_list,
          exp3.model_shift_distance_list
        ).save_to_csv(RECOURSENUM, THRESHOLD, POSITIVE_RATIO, COSTWEIGHT, DATASET, current_time,DIRECTORY)