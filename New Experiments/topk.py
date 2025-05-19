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

from Models.logisticRegression import LogisticRegression, training
from Models.recourseGradient_simple import recourse
from Config.config import train, test, sample, model, dataset, POSITIVE_RATIO # modified parameters for observations
from Dataset.makeDataset import Dataset


current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")

# modified parameters for observations
THRESHOLD = 0.7           #0.5 0.7 0.9
RECOURSENUM = 0.5          #0.2 0.5 0.7
COSTWEIGHT = 'uniform'     #uniform log extreme extreme2
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

    def update(self, model: nn.Module, train: Dataset, sample: Dataset, recoursedFail, recoursedSuccess):
        print("round: ",self.round)
        self.round += 1


        self.model_params = deepcopy(self.model.state_dict())
        
        if self.round != 1:
            #randomly select from self.sample with size of train and label it with model
            self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'mixed')

            # find training data with label 0 and select RECOURSENUM of them
            data, labels = self.train.x, self.train.y
            label_0_indices = pt.where(labels == 0)[0]
            shuffled_indices = pt.randperm(len(label_0_indices))
            label_0_indices = label_0_indices[shuffled_indices]
            num_samples = math.floor(len(label_0_indices) * RECOURSENUM)
            selected_indices = label_0_indices[:num_samples]


            # perform recourse on the selected subset
            selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
            recourse_weight = getWeights(self.train.x.shape[1], COSTWEIGHT) 
            recoursed, action = recourse(
                self.model,
                selected_subset,
                100,
                recourse_weight,
                loss_list=[],
                threshold=THRESHOLD,
                cost_list=self.avgRecourseCost_list,
                q3RecourseCost=self.q3RecourseCost,
                recourseModelLossList=self.recourseModelLossList,
                isNew = pt.tensor([]) if isNewList.numel() == 0 else isNewList[selected_indices],
                new_cost_list=self.avgNewRecourseCostList,
                original_cost_list=self.avgOriginalRecourseCostList
            )
            # print("recourse action", action)
            recoursed_data = recoursed.x
            self.train.x[selected_indices] = recoursed_data
            
            # update the labels of D using topk method
            with pt.no_grad():
                y_prob_all: pt.Tensor = self.model(self.train.x)
            sorted_indices = pt.argsort(y_prob_all[:, 0], dim=0, descending=True)
            cutoff_index = int(len(sorted_indices) * POSITIVE_RATIO)
            mask = pt.zeros_like(y_prob_all)
            mask[sorted_indices[:cutoff_index]] = 1
            self.train.y = mask.float().squeeze(1)
            
            # calculate the average score of the model on the last training data
            score = self.train.x @ self.model.linear.weight.T + self.model.linear.bias
            avg_score1 = score.mean()

            # train the model with the updated dataset
            training(self.model, self.train, 10, self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)
            
        #calculate metrics: ========================================================================

        #calculate higher standard (model output before sigmoid) based on last train data
        if self.historyTrainList != []:
            last_data = self.historyTrainList[-1]
            last_data.x = self.train.x
            score = last_data.x @ self.model.linear.weight.T + self.model.linear.bias
            avg_score2 = score.mean()
            
            self.avg_score_on_last_train.append(avg_score1.item() - avg_score2.item())
        else:
            self.avg_score_on_last_train.append(0)
            
        #calculate short term accuracy
        current_data = Dataset(self.train.x, self.train.y)
        self.historyTrainList.append(current_data)
        with pt.no_grad():
            y_prob_test: pt.Tensor = self.model(self.test.x)
        y_prob_test = y_prob_test.squeeze(1)
        y_pred_test = (y_prob_test > 0.5).float()
        self.test.y = y_pred_test
        current_test = Dataset(self.test.x, self.test.y)
        self.historyTestList.append(current_test)
        self.overall_acc_list.append(self.calculate_AA(self.model, self.historyTestList, 7, "typical"))
        self.balanced_acc_list.append(self.calculate_AA(self.model, self.historyTestList, 7, "balanced"))
        self.cohen_acc_list.append(self.calculate_AA(self.model, self.historyTestList, 7, "cohen_kappa"))

        #calculate short term accuracy without recourse
        if self.round != 1:
            all_indices = pt.arange(self.train.x.size(0))  # All possible indices
            mask = pt.ones(self.train.x.size(0), dtype=bool)
            mask[selected_indices] = False  # Mask out the selected indices
            

            current_data_without_recourse = Dataset(self.train.x[mask], self.train.y[mask])
            self.historyTrainList_withoutRecourse.append(current_data_without_recourse)
            self.overall_acc_list_withoutRecourse.append(self.calculate_AA(self.model, self.historyTrainList_withoutRecourse, 7))
        
        else:
            self.historyTrainList_withoutRecourse.append(current_data)
            self.overall_acc_list_withoutRecourse.append(self.overall_acc_list[-1])

        if self.round != 1:
            #calculate ftr
            fail_positions = pt.where(self.train.y[selected_indices] == 0)[0]
            success_positions = pt.where(self.train.y[selected_indices] == 1)[0]
            self.recoursedFail = selected_indices[fail_positions]
            self.recoursedSuccess = selected_indices[success_positions]
            recourseFailCnt = fail_positions.shape[0] if fail_positions.shape[0] > 0 else 0
            recourseFailRate = recourseFailCnt / len(self.train.y[selected_indices])
            self.failToRecourse.append(recourseFailRate)
            

            if isNewList.numel() != 0:
                #calculate ftr_old
                new_indices = isNewList[selected_indices]
                old_selected_indices = selected_indices[new_indices == False]
                recourseFailCnt_old = pt.where(self.train.y[old_selected_indices] == 0)[0].shape[0]
                recourseFailRate_old = recourseFailCnt_old / len(self.train.y[old_selected_indices])
                self.failToRecourse_old.append(recourseFailRate_old)
                
                #calculate ftr_new
                new_selected_indices = selected_indices[new_indices == True]
                # print(f"new_selected_indices: {len(new_selected_indices)}")
                recourseFailCnt_new = pt.where(self.train.y[new_selected_indices] == 0)[0].shape[0]
                recourseFailRate_new = recourseFailCnt_new / len(self.train.y[new_selected_indices])
                self.failToRecourse_new.append(recourseFailRate_new)
                
            else:
                self.failToRecourse_old.append(0)
                self.failToRecourse_new.append(0)

        else:
            self.failToRecourse.append(0)
            self.failToRecourse_old.append(0)
            self.failToRecourse_new.append(0)

        #jsd is calculated in helper.py already

        #calculate t_rate
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(test.x)
        #calculate the ratio of 1s and 0s in the test data
        num_ones = pt.where(y_prob > 0.5)[0].shape[0]
        num_zeros = len(y_prob) - num_ones
        t_rate = num_ones / num_zeros
        self.t_rate_list.append(t_rate)
        

        #calculate model shift distance
        last_model_params = self.model_params
        current_model_params = self.model.state_dict()
        shift_distance = pt.norm(
            pt.cat([pt.flatten(last_model_params[key] - current_model_params[key])
                for key in last_model_params.keys()]), p=2
        )
        self.model_shift_distance_list.append(shift_distance)

        
        # calculate average entropy of the model
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(self.train.x)
        y_prob = y_prob.squeeze(1)
        y_prob = pt.clamp(y_prob, min=1e-7, max=1 - 1e-7)  # Avoid log(0)
        entropy = -pt.mean(y_prob * pt.log(y_prob) + (1 - y_prob) * pt.log(1 - y_prob))
        self.entropy_list.append(entropy.item())


        # calculate average score before sigmoid
        score = self.test.x @ self.model.linear.weight.T + self.model.linear.bias
        avg_score = score.mean()
        self.avg_score_list.append(avg_score.item())
        print("====================================================")


exp2 = Exp2(model, pca, train, test, sample)
exp2.save_directory = DIRECTORY
ani1 = exp2.animate_all(101)
current_time = datetime.datetime.now().strftime("%d-%H-%M")

ani1.save(os.path.join(DIRECTORY, f"{RECOURSENUM}_{THRESHOLD}_{POSITIVE_RATIO}_{COSTWEIGHT}_{DATASET}_{current_time}.mp4"))
exp2.overall_acc_list = exp2.overall_acc_list
exp2.draw_avgRecourseCost()
exp2.plot_jsd()
exp2.draw_Fail_to_Recourse()
exp2.plot_aac()
exp2.plot_t_rate()
exp2.plot_model_shift()
exp2.plot_model_shift_w_diff_cost()
exp2.plot_feature_ranking()

# save to csv
FileSaver(exp2.failToRecourse, 
          exp2.overall_acc_list, 
          exp2.jsd_list, 
          exp2.avgRecourseCost_list, 
          exp2.avgNewRecourseCostList,
          exp2.avgOriginalRecourseCostList,
          exp2.t_rate_list,
          exp2.model_shift_distance_list,
          exp2.failToRecourse_old,
          exp2.failToRecourse_new,
          exp2.entropy_list,
          exp2.avg_score_list,
          exp2.overall_acc_list_withoutRecourse,
          exp2.avg_score_on_last_train,
          exp2.balanced_acc_list,
          exp2.cohen_acc_list,
        ).save_to_csv(RECOURSENUM, THRESHOLD, POSITIVE_RATIO, COSTWEIGHT, DATASET, current_time, DIRECTORY)