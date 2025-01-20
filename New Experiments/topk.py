import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display
import math


import os
import sys
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Experiment_Helper.auxiliary import getWeights, update_train_data, FileSaver

# from Models.MLP import MLP, training
from Models.logisticRegression import LogisticRegression, training

from Models.recourseGradient import recourse

from Config.config import train, test, sample, model, dataset, POSITIVE_RATIO # modified parameters for observations
# from Config.MLP_config import train, test, sample, model, dataset, POSITIVE_RATIO # modified parameters for observations

from Dataset.makeDataset import Dataset


current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")

# modified parameters for observations
THRESHOLD = 0.5            #0.5 0.7 0.9
RECOURSENUM = 0.5          #0.2 0.5 0.7
COSTWEIGHT = 'uniform'     #uniform log
DATASET = dataset

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
        
        #save model parameters
        self.model_params = deepcopy(self.model.state_dict())

        #randomly select from self.sample with size of train and label it with model
        self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'mixed')
        print(f"isNewList: {isNewList}")
        print(f"isNewList.shape: {isNewList.shape}")

        # find training data with label 0 and select 1/2 of them
        data, labels = self.train.x, self.train.y
        label_0_indices = pt.where(labels == 0)[0]
        shuffled_indices = pt.randperm(len(label_0_indices))
        label_0_indices = label_0_indices[shuffled_indices]
        num_samples = math.floor(len(label_0_indices) * RECOURSENUM)
        selected_indices = label_0_indices[:num_samples]
        print(f"selected_indice: {selected_indices}")
        print(f"isNewList[selected_indices]: {isNewList[selected_indices]}")

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

        #calculate metrics: ========================================================================
        #calculate short term accuracy
        current_data = Dataset(self.train.x, self.train.y)
        self.historyTrainList.append(current_data)
        self.overall_acc_list.append(self.calculate_AA(self.model, self.historyTrainList, 4))

        #calculate ftr
        recourseFailCnt = pt.where(self.train.y[selected_indices] == 0)[0].shape[0]
        recourseFailRate = recourseFailCnt / len(self.train.y[selected_indices])
        self.failToRecourse.append(recourseFailRate)
        print("recourseFailRate: ",recourseFailRate)

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



exp2 = Exp2(model, pca, train, test, sample)
exp2.save_directory = DIRECTORY
ani1 = exp2.animate_all(100)
current_time = datetime.datetime.now().strftime("%d-%H-%M")
ani1.save(os.path.join(DIRECTORY, f"{RECOURSENUM}_{THRESHOLD}_{POSITIVE_RATIO}_{COSTWEIGHT}_{DATASET}_{current_time}.mp4"))
exp2.overall_acc_list = exp2.overall_acc_list[3:]
exp2.draw_avgRecourseCost()
exp2.plot_jsd()
exp2.draw_Fail_to_Recourse()
exp2.plot_aac()
exp2.plot_t_rate()
exp2.plot_model_shift()

# save to csv
FileSaver(exp2.failToRecourse, 
          exp2.overall_acc_list, 
          exp2.jsd_list, 
          exp2.avgRecourseCost_list, 
          exp2.avgNewRecourseCostList, 
          exp2.avgOriginalRecourseCostList,
          exp2.t_rate_list,
          exp2.model_shift_distance_list
        ).save_to_csv(RECOURSENUM, THRESHOLD, POSITIVE_RATIO, COSTWEIGHT, DATASET, current_time, DIRECTORY)