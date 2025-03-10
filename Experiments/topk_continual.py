import torch as pt
from torch import nn, optim
from copy import deepcopy
import numpy as np

import csv
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Models.logisticRegression import LogisticRegression, training
from Models.synapticIntelligence import continual_training
from Models.recourseGradient import recourse
from Config.continual_config import train, test, sample, si, POSITIVE_RATIO
from Dataset.makeDataset import Dataset

LAMBDA = 0.001

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")
RESULT_DIR = f"Result\{current_file_name}.csv"

try:
    os.makedirs(DIRECTORY, exist_ok=True)
    print(f"Folder '{DIRECTORY}' is ready.")
except Exception as e:
    print(f"An error occurred: {e}")

class Example9_Continual_Learning(Helper):
    '''
    Update Method Steps:
    1. Selects a random subset of `sample` with a approximately size of `train.size * 0.02`.
    2. Performs recourse on the selected subset of 'sample' while preserving their original labels.
    3. Replaces a corresponding part of the training set with the updated samples.
    4. Refits the model with the modified training data.
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        #get model parameters before model updated
        modelParams = list(model.parameters())
        weights = deepcopy(modelParams[0].data.reshape(-1))
        bias = deepcopy(modelParams[1].data)
        # print("Before update: ")
        # print(modelParams)
        # print(weights,bias)


        size = train.x.shape[0] // 4
        i = np.random.choice(sample.x.shape[0], size, False)
        x = sample.x[i]
        # print("x:",x)

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x)

        # print("predict: ",y_prob.data)
        y_pred = y_prob.flatten() < 0.5
        # print("~y_pred : ",y_pred)
        sub_sample = Dataset(x[y_pred], pt.ones((y_pred.count_nonzero(), 1)))

        # recourse(model, sub_sample, 10,weight,loss_list=[])
        recourse(self.model, sub_sample, 10, pt.from_numpy(np.ones(train.x.shape[1])) / train.x.shape[1], threshold=0.6, cost_list = self.avgRecourseCost_list, q3RecourseCost = self.q3RecourseCost, recourseModelLossList = self.recourseModelLossList)

        x[y_pred] = sub_sample.x


        j = np.random.choice(train.x.shape[0], size, False)
        self.train.x[j] = x
        with pt.no_grad():
            self.train.y[j] = (self.model(x).flatten() > 0.5).float()

        # print("train.y[j, 0] : ",train.y[j, 0])
        # with pt.no_grad():
        #     tensor_list = model(train.x[j]).tolist()
        #     # Format the numbers to float
        #     formatted_list = [[float(f"{num:.6f}") for num in sublist] for sublist in tensor_list]
        #     # Print the formatted list
        #     for sublist in formatted_list:
        #         print(sublist)

        # index = y_prob.flatten() > 0.5
        # k = round(len(index) * 0.4)
        # index = y_prob[index].numpy().argpartition(k)
        # train.y[]


        # fail to recourse on Label
        with pt.no_grad():
            y_prob_l: pt.Tensor = self.model(x[y_pred])

        
        recourseFailCnt = len(y_prob_l[y_prob_l < 0.5])
        print("recourseFailCnt",recourseFailCnt, "len(x[y_pred])",len(x[y_pred]))
        if len(x[y_pred]) == 0:
            recourseFailRate = -1
        else:
            recourseFailRate = recourseFailCnt / len(x[y_pred])
        # print("recourseFailRate : ",recourseFailRate)
        self.failToRecourseOnLabel.append(recourseFailRate)

        with pt.no_grad():
          y_prob_all: pt.Tensor = self.model(self.train.x)

        # top 50% 
        sorted_indices = pt.argsort(y_prob_all[:, 0], dim=0, descending=True)
        cutoff_index = int(len(sorted_indices) * POSITIVE_RATIO)
        # print("sorted_indices", sorted_indices)
        mask = pt.zeros_like(y_prob_all)
        mask[sorted_indices[:cutoff_index]] = 1
        self.train.y = mask.float().squeeze()
        

        val_data = Dataset(self.train.x[j], self.train.y[j])
        self.validation_list.append(val_data)
        sample_model = LogisticRegression(val_data.x.shape[1], 1)
        sample_model.train()
        print(val_data.x.shape, val_data.y.shape)
        training(sample_model, val_data, 30, self.test)
        self.Aj_tide_list.append(self.calculate_accuracy(sample_model(val_data.x), val_data.y))

        #紀錄新增進來的sample資料
        self.addEFTDataFrame(j)

        if(self.jsd_list == []):
            continual_training(self.si, self.train, 50, lambda_ = 0)
        else:
            continual_training(self.si, self.train, 50, lambda_ = 0.0001/(self.jsd_list[-1]))

        self.cnt += 1
        self.si.update_omega(self.train, nn.BCELoss())
        self.si.consolidate(3)

        # calculate the overall accuracy
        self.overall_acc_list.append(self.calculate_AA(self.model, self.validation_list, 3))
        # evaluate memory stability
        self.memory_stability_list.append(self.calculate_BWT(self.model, self.validation_list, self.Ajj_performance_list, 3))
        # evaluate learning plasticity
        self.memory_plasticity_list.append(self.calculate_FWT(self.Ajj_performance_list, self.Aj_tide_list, 3))


        #紀錄Fail_to_Recourse on Model
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x[y_pred])

        # print("x[y_pred] : ",x[y_pred])
        # print("after model update:")
        # print("y_prob:",y_prob)
        # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])
        recourseFailCnt = len(y_prob[y_prob < 0.5])
        # print("recourseFailCnt",recourseFailCnt)
        if len(x[y_pred]) == 0:
            recourseFailRate = 0
        else:
            recourseFailRate = recourseFailCnt / len(x[y_pred])
        # print("recourseFailRate : ",recourseFailRate)
        self.failToRecourseOnModel.append(recourseFailRate)

        self.EFTdataframe = self.EFTdataframe.assign(updateRounds = self.EFTdataframe['updateRounds'] + 1)
        self.round = self.round + 1


        #updated model predict the data with new sample
        data = np.vstack(self.EFTdataframe['x'])
        with pt.no_grad():
            y_pred = self.model(pt.tensor(data,dtype = pt.float))

        #set prob 0.5 as threshold
        predictValue = deepcopy(y_pred.data)
        predictValue[predictValue > 0.5] = 1.0
        predictValue[predictValue < 0.5] = 0.0
        predictValue = predictValue.numpy().T.reshape(-1)
        # print("predictValue: ",predictValue)
        # print("data type: ",type(predictValue))

        #store the updated model predict value
        for i in self.EFTdataframe.index:
            self.EFTdataframe.at[i,'Predict'].append(predictValue[i])
         # self.EFTdataframe.loc[predictValue.numpy().T.reshape(-1) != self.EFTdataframe['Predict'],['flip_times']] += 1

         #check whether the output flip out
        for i in self.EFTdataframe.index:
            predictLength = len(self.EFTdataframe.at[i,'Predict'])
            # if predictLength > 1 and (self.EFTdataframe.at[i,'Predict'][predictLength - 2] != self.EFTdataframe.at[i,'Predict'][predictLength - 1]):
            if predictLength > 1 and (self.EFTdataframe.at[i,'Predict'][-2] != self.EFTdataframe.at[i,'Predict'][-1]):
                # self.EFTdataframe.loc[(self.EFTdataframe['rounds'] - self.EFTdataframe['startRounds'] > 1) and self.EFTdataframe['Predict'][self.round - 2] != self.EFTdataframe['Predict'][self.round - 1] ,['flip_times']] += 1
                self.EFTdataframe.at[i,'flip_times'] += 1

        #update EFP values
        self.EFTdataframe.loc[(self.EFTdataframe['updateRounds'] - 1) > 0,['EFT']] = self.EFTdataframe['flip_times'] / (self.EFTdataframe['updateRounds'] - 1)

        for i in self.EFTdataframe.index:
            if len(self.EFTdataframe.at[i,'Predict']) > 1:
                # self.EFTdataframe.at[i,'EFTList'].append(self.EFTdataframe.at[i,'flip_times'] / self.EFTdataframe.at[i,'rounds'])
                self.EFTdataframe.at[i,'EFTList'].append(self.EFTdataframe.at[i,'EFT'])

        # display(self.EFTdataframe)
        #calculate the PDt
        modelParams = list(self.model.parameters())
        modelParameter = np.concatenate((weights,bias))
        resultParameter = np.concatenate((modelParams[0].data.reshape(-1),modelParams[1].data))
        # print("Before update: ")
        # print(weights,bias)
        # print("After update: ")
        # print(modelParams[0].data.reshape(-1),modelParams[1].data)

        parameterL2 = np.linalg.norm(resultParameter - modelParameter)

        self.PDt.append(parameterL2)

weight = pt.from_numpy(np.ones(train.x.shape[1]))
# print(train.x)
# print(train.y)

ex9 = Example9_Continual_Learning(si.model, pca, train, test, sample)
ex9.si = si
# ani1 = ex1.animate_all(240)
ex9.save_directory = DIRECTORY
ROUNDS = 80
ani9 = ex9.animate_all(ROUNDS, inplace = True)
ani9.save(os.path.join(DIRECTORY, "ex9.mp4"))

# ex1.draw_PDt()
# ex9.draw_PDt()
# ex9.draw_EFT(ROUNDS)
# ex9.draw_R20_EFT(ROUNDS,8)
# ex9.draw_R20_EFT(ROUNDS,10)

# ex1.draw_EFT(240)
# ex1.draw_R20_EFT(240,23)
# ex1.draw_R20_EFT(240,40)
# ex1.draw_R20_EFT(240,58)
ex9.draw_Fail_to_Recourse_on_Model()
ex9.draw_Fail_to_Recourse_on_Label()
# display(ex9.EFTdataframe)
ex9.plot_matricsA()
ex9.plot_Ajj()
ex9.plot_jsd()
print(ex9.testacc)

column_name = f'{current_file_name}_k-{POSITIVE_RATIO}_c-{LAMBDA}'
if os.path.exists(RESULT_DIR):
    # Read the existing CSV file
    df = pd.read_csv(RESULT_DIR)

    # Append the new column (if it doesn't already exist)
    df[column_name] = ex9.failToRecourseOnModel

else:
    # Create a new DataFrame with the new column
    df = pd.DataFrame({column_name: ex9.failToRecourseOnModel})

# Save the DataFrame back to the CSV file
df.to_csv(RESULT_DIR, index=False)