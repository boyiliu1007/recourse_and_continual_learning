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

#主要都寫在裡面，其他的還沒加
class Example8(Helper):
    '''
    Update Method Steps:
    1. Selects a random subset of `sample` with a size of `train.size * 0.02`.
    2. Performs recourse on the selected samples with binomial distributed destination.
    3. Randomly relabels samples based on the model's probability scores.
    4. Replaces a corresponding part of the training set with the updated samples.
    5. Refits the model with the modified training data.
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        modelParams = list(model.parameters())
        weights = deepcopy(modelParams[0].data.reshape(-1))
        bias = deepcopy(modelParams[1].data)
        # Selects a random subset of `sample` with a size of `train.size * 0.02`
        size = train.x.shape[0] // 10
        i = np.random.choice(sample.x.shape[0], size, False)
        x = sample.x[i]

        with pt.no_grad():
            y_prob: pt.Tensor = model(x)
        # print("y_prob :",y_prob)
        y_pred = y_prob.flatten() < 0.5
        # print("y_pred :",y_pred)

        # Performs recourse on the selected samples with binomial distributed destination.
        binomialData = np.random.binomial(1,0.6,np.count_nonzero(y_pred)).astype(bool)
        # print("binomial data: ",binomialData)
        y_pred[y_pred == True] = pt.from_numpy(binomialData)
        print("binomial y_pred",y_pred)
        # print("x : ",x)
        # print("x[y_pred] : ",x[y_pred])
        sub_sample = Dataset(x[y_pred], pt.full((y_pred.count_nonzero(), 1), 0.6))

        recourse(model, sub_sample, 5)

        x[y_pred] = sub_sample.x

        #Randomly relabels samples based on the model's probability scores.
        j = np.random.choice(train.x.shape[0], size, False)

        with pt.no_grad():
            y_prob: pt.Tensor = model(x)

        # print("y_prob : ",y_prob)
        # print("y_prob[y_pred] :",y_prob[y_pred])

        # Replaces a corresponding part of the training set with the updated samples.
        train.x[j] = x
        train.y[j] = (pt.rand_like(y_prob) > y_prob).float()

        val_data = Dataset(train.x[j], train.y[j])
        self.validation_list.append(val_data)
        sample_model = LogisticRegression(val_data.x.shape[1], 1)
        sample_model.train()
        training(sample_model, val_data, 30)
        self.Aj_tide_list.append(self.calculate_accuracy(sample_model(val_data.x), val_data.y))

        # #紀錄新增進來的sample資料
        self.addEFTDataFrame(j)

        training(model, train, 20)

        # calculate the overall accuracy
        self.overall_acc_list.append(self.calculate_AA(model, self.validation_list))
        # evaluate memory stability
        self.memory_stability_list.append(self.calculate_BWT(model, self.validation_list, self.Ajj_performance_list))
        # evaluate memory plasticity
        self.memory_plasticity_list.append(self.calculate_FWT(self.Ajj_performance_list, self.Aj_tide_list))

        #紀錄Fail_to_Recourse
        if len(x[y_pred]) > 0:
            with pt.no_grad():
                y_prob: pt.Tensor = model(x[y_pred])

            # print("x[y_pred] : ",x[y_pred])
            # print("after model update:")
            # print("y_prob:",y_prob)
            # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])
            recourseFailCnt = len(y_prob[y_prob < 0.5])
            # print("recourseFailCnt",recourseFailCnt)
            recourseFailRate = recourseFailCnt / len(x[y_pred])
            # print("recourseFailRate : ",recourseFailRate)
            self.failToRecourse.append(recourseFailRate)
        else:
            print("no Recourse:")
            self.failToRecourse.append(0)



        self.EFTdataframe = self.EFTdataframe.assign(updateRounds = self.EFTdataframe['updateRounds'] + 1)
        self.round = self.round + 1

        #updated model predict the data with new sample
        data = np.vstack(self.EFTdataframe['x'])
        with pt.no_grad():
            y_pred = model(pt.tensor(data,dtype = pt.float))
        # print(y_pred)
        predictValue = deepcopy(y_pred.data)
        predictValue[predictValue > 0.5] = 1.0
        predictValue[predictValue < 0.5] = 0.0
        predictValue = predictValue.numpy().T.reshape(-1)
        # print("predictValue: ",predictValue)
        # print("data type: ",type(predictValue))
        for i in self.EFTdataframe.index:
            # for j in predictValue.numpy().T.reshape(-1):
            self.EFTdataframe.at[i,'Predict'].append(predictValue[i])


        # self.EFTdataframe.loc[predictValue.numpy().T.reshape(-1) != self.EFTdataframe['Predict'],['flip_times']] += 1
        for i in self.EFTdataframe.index:
            predictLength = len(self.EFTdataframe.at[i,'Predict'])
            # if predictLength > 1 and (self.EFTdataframe.at[i,'Predict'][predictLength - 2] != self.EFTdataframe.at[i,'Predict'][predictLength - 1]):
            if predictLength > 1 and (self.EFTdataframe.at[i,'Predict'][-2] != self.EFTdataframe.at[i,'Predict'][-1]):
                # self.EFTdataframe.loc[(self.EFTdataframe['rounds'] - self.EFTdataframe['startRounds'] > 1) and self.EFTdataframe['Predict'][self.round - 2] != self.EFTdataframe['Predict'][self.round - 1] ,['flip_times']] += 1
                self.EFTdataframe.at[i,'flip_times'] += 1

        # self.EFTdataframe = self.EFTdataframe.assign(Predict = predictValue)

        #update EFP values
        # self.EFTdataframe['EFT'] = self.EFTdataframe['flip_times'] / self.EFTdataframe['rounds']
        self.EFTdataframe.loc[(self.EFTdataframe['updateRounds'] - 1) > 0,['EFT']] = self.EFTdataframe['flip_times'] / (self.EFTdataframe['updateRounds'] - 1)

        for i in self.EFTdataframe.index:
            if len(self.EFTdataframe.at[i,'Predict']) > 1:
                # self.EFTdataframe.at[i,'EFTList'].append(self.EFTdataframe.at[i,'flip_times'] / self.EFTdataframe.at[i,'rounds'])
                self.EFTdataframe.at[i,'EFTList'].append(self.EFTdataframe.at[i,'EFT'])
        # else:
        #     self.EFTdataframe = self.EFTdataframe.assign(Predict = predictValue)
        # display(self.EFTdataframe[self.EFTdataframe['flip_times'] > 0])
        display(self.EFTdataframe)

        #calculate the PDt
        modelParams = list(model.parameters())
        modelParameter = np.concatenate((weights,bias))
        resultParameter = np.concatenate((modelParams[0].data.reshape(-1),modelParams[1].data))

        parameterL2 = np.linalg.norm(resultParameter - modelParameter)

        self.PDt.append(parameterL2)



ex8 = Example8(model, pca, train, test, sample)
ex8.save_directory = DIRECTORY
ani8 = ex8.animate_all(240)
ani8.save(os.path.join(DIRECTORY, "ex8.gif"))

ex8.draw_PDt()
ex8.draw_EFT(240)
ex8.draw_R20_EFT(240,23)
ex8.draw_R20_EFT(240,40)
ex8.draw_R20_EFT(240,58)
print(ex8.failToRecourse)
ex8.draw_Fail_to_Recourse()
display(ex8.EFTdataframe)
ex8.plot_matricsA()