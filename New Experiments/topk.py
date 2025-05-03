import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np
from IPython.display import display
import math
from sklearn.ensemble import RandomForestClassifier

import os
import sys
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Experiment_Helper.auxiliary import getWeights, update_train_data, FileSaver

# from Models.MLP import MLP, training
from Models.logisticRegression import LogisticRegression, training

from Models.recourseGradient_simple import recourse

from Config.config import train, test, sample, model, dataset, POSITIVE_RATIO # modified parameters for observations
# from Config.MLP_config import train, test, sample, model, dataset, POSITIVE_RATIO # modified parameters for observations

from Dataset.makeDataset import Dataset


current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")

# modified parameters for observations
THRESHOLD = 0.7           #0.5 0.7 0.9
RECOURSENUM = 0.5          #0.2 0.5 0.7
COSTWEIGHT = 'extreme2'     #uniform log extreme
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

        # if self.round == 1:
        #     with pt.no_grad():  # Disable gradient tracking
        #         self.model.linear.weight.copy_(pt.full_like(self.model.linear.weight, 0.1))
        #         self.model.linear.bias.copy_(pt.full_like(self.model.linear.bias, 0))
        #     print("initial model params", self.model.state_dict())
        #save model parameters
        if(self.round == 2): # here round 2 is the first round of recourse since round 1 is the initial model
            self.first_model_params = deepcopy(self.model_params)
            print("first model params", self.first_model_params)
        self.model_params = deepcopy(self.model.state_dict())
        
        if self.round != 1:
            #randomly select from self.sample with size of train and label it with model
            self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'none')

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

            # feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(train.x, train.y.ravel())  # Make sure y is 1D

            # Get feature importances
            importances = rf.feature_importances_

            ranked_features = np.argsort(-importances)
            feature_to_rank = np.zeros_like(ranked_features)
            for rank, feature_idx in enumerate(ranked_features):
                feature_to_rank[feature_idx] = rank

            # Now sum up the ranks of dimensions 0 ~ 5
            # feature_rank_sum = 0
            # for feature_idx in range(0, 5):  # feature 0, 1, 2, 3, 4, 5
            #     feature_rank_sum += feature_to_rank[feature_idx]
            # print("feature rank sum", feature_rank_sum/5)
            # self.low_cost_feature_ranking.append(feature_rank_sum/5)
            important_feature_rank_sum = 0
            important_feature_rank_sum += feature_to_rank[16]+feature_to_rank[18]+feature_to_rank[7]+feature_to_rank[13]
            important_feature_rank_sum /= 4
            print("important feature rank sum", important_feature_rank_sum)
            self.important_feature_ranking.append(important_feature_rank_sum)
            
            # print the sum of train.x feature wise
            print("sum of train.x feature wise", pt.sum(self.train.x, dim=0))

            # train the model with the updated dataset
            training(self.model, self.train, 50, self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)
            print("Classifier weights:", self.model.linear.weight.data)
        #calculate metrics: ========================================================================

        #calculate short term accuracy
        current_data = Dataset(self.train.x, self.train.y)
        self.historyTrainList.append(current_data)
        self.overall_acc_list.append(self.calculate_AA(self.model, self.historyTrainList, 7))

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
        shift_distance = sum(
            pt.norm(last_model_params[key] - current_model_params[key], p=2) ** 2
            for key in last_model_params.keys()
        ).sqrt()
        self.model_shift_distance_list.append(shift_distance)

        
        #calculate cosine similarity
        cosine_similarity = sum(
            pt.nn.functional.cosine_similarity(
                last_model_params[key].view(-1),
                current_model_params[key].view(-1),
                dim=0  # make sure to compare full vectors
            )
            for key in last_model_params.keys() if 'weight' in key
        )
        # print("last model with current model cosine", cosine_similarity.item())

        if (self.first_model_params != None):
            cosine_similarity = sum(
                pt.nn.functional.cosine_similarity(
                    self.first_model_params[key].view(-1),
                    current_model_params[key].view(-1),
                    dim=0  # make sure to compare full vectors
                )
                for key in last_model_params.keys() if 'weight' in key
            )
            # print("first model with current model cosine", cosine_similarity.item())

        # calculate model shift on each feature
        model_shift_per_feature = {}
        for key in last_model_params.keys():
            if 'weight' in key:
                diff = last_model_params[key] - current_model_params[key]  
                shift_per_feature = pt.norm(diff, p=2, dim=0) 
                model_shift_per_feature[key] = shift_per_feature.tolist()

        # print("Model shift per feature:", model_shift_per_feature)
        low_cost_model_shift = abs(model_shift_per_feature['linear.weight'][0]) + abs(model_shift_per_feature['linear.weight'][1]) + abs(model_shift_per_feature['linear.weight'][2]) + abs(model_shift_per_feature['linear.weight'][3]) + abs(model_shift_per_feature['linear.weight'][4])
        low_cost_model_shift /= 5
        high_cost_model_shift = 0
        for i in range(5, len(model_shift_per_feature['linear.weight'])):
            high_cost_model_shift += abs(model_shift_per_feature['linear.weight'][i])
        high_cost_model_shift /= len(model_shift_per_feature['linear.weight']) - 5

        # print("low avg cost shift", low_cost_model_shift)
        # print("high avg cost shift", high_cost_model_shift)
        self.low_cost_model_shift_list.append(low_cost_model_shift)
        self.high_cost_model_shift_list.append(high_cost_model_shift)


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
# FileSaver(exp2.failToRecourse, 
#           exp2.overall_acc_list, 
#           exp2.jsd_list, 
#           exp2.avgRecourseCost_list, 
#           exp2.avgNewRecourseCostList, 
#           exp2.avgOriginalRecourseCostList,
#           exp2.t_rate_list,
#           exp2.model_shift_distance_list,
#           exp2.failToRecourse_old,
#           exp2.failToRecourse_new
#         ).save_to_csv(RECOURSENUM, THRESHOLD, POSITIVE_RATIO, COSTWEIGHT, DATASET, current_time, DIRECTORY)