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
from Config.MLP_config import MLP_model
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

#binomial Recourse distribution    Prob = 0.01
class Example1(Helper):
    '''
    Update Method Steps:
    1. Selects a random subset of `sample` with a approximately size of `train.size * 0.02`.
    2. Performs recourse on the selected subset of 'sample' while preserving their original labels.
    3. Replaces a corresponding part of the training set with the updated samples.
    4. Refits the model with the modified training data.
    '''

    #add first k              k maybe 40% 80% 100%

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
            y_prob: pt.Tensor = model(x)

        # print("predict: ",y_prob.data)
        y_pred = y_prob.flatten() < 0.5
        # print("y_pred: ",y_pred)
        # print("~y_pred: ",(~y_pred).float())

        # Performs recourse on the selected samples with binomial distributed destination.
        binomialData = np.random.binomial(1,BinomialProb,np.count_nonzero(y_pred)).astype(bool)
        # print("binomial data: ",binomialData)
        # y_pred[y_pred == True] = pt.from_numpy(binomialData)
        recourseIndex = deepcopy(y_pred)
        # print("before binomial",recourseIndex)
        recourseIndex[recourseIndex == True] = pt.from_numpy(binomialData)
        # print("binomial y_pred",y_pred)
        print("binomial recourseIndex",recourseIndex)
        sub_sample = Dataset(x[recourseIndex], pt.full((recourseIndex.count_nonzero(), 1),1.0))

        # print("do the recourse")
        # print("len subsample",len(sub_sample))
        # recourse(model, sub_sample, 10,weight,loss_list=[])
        if len(sub_sample) > 0:
            recourse(model, sub_sample, 120,threshold=0.9,loss_list=[])

            # print("sub_sample x:",sub_sample.x)
            # print("sub_sample y:",sub_sample.y)
            # test = deepcopy(x)
            # print("x : ",test)

            x[recourseIndex] = sub_sample.x

            # print("after recourse")
            # print("check eq :",pt.eq(test,x))
        else:
            a = deepcopy(x)
            b = Dataset(a[y_pred], pt.full((y_pred.count_nonzero(), 1),1.0))
            print("b.x : ",len(b.x))
            print("b.y : ",len(b.y))
            recourse(model, b, 120,threshold=0.9,loss_list=[])
            a[y_pred] = b.x
            with pt.no_grad():
                y_prob: pt.Tensor = model(a[y_pred])
            print("Before model update : ")
            print("y_prob:",y_prob)
            print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])

        j = np.random.choice(train.x.shape[0], size, False)
        train.x[j] = x
        train.y[j, 0] = (~y_pred).float()

        # index = y_prob.flatten() > 0.5
        # k = round(len(index) * 0.4)
        # index = y_prob[index].numpy().argpartition(k)
        # train.y[]
        # with pt.no_grad():
        #     y_prob: pt.Tensor = model(a[y_pred])
        # print("Before model update : ")
        # print("y_prob:",y_prob)
        # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])

        val_data = Dataset(train.x[j], train.y[j])
        self.validation_list.append(val_data)
        sample_model = LogisticRegression(val_data.x.shape[1], 1)
        sample_model.train()
        training(sample_model, val_data, 30)
        self.Aj_tide_list.append(self.calculate_accuracy(sample_model(val_data.x), val_data.y))

        #紀錄新增進來的sample資料
        self.addEFTDataFrame(j)


        training(model, train, 50)

        # calculate the overall accuracy
        self.overall_acc_list.append(self.calculate_AA(model, self.validation_list))
        # evaluate memory stability
        self.memory_stability_list.append(self.calculate_BWT(model, self.validation_list, self.Ajj_performance_list))
        # evaluate memory plasticity
        self.memory_plasticity_list.append(self.calculate_FWT(self.Ajj_performance_list, self.Aj_tide_list))


        #紀錄Fail_to_Recourse
        if len(x[recourseIndex]) > 0:
            with pt.no_grad():
                y_prob: pt.Tensor = model(x[recourseIndex])

            # print("x[recourseIndex] : ",x[recourseIndex])
            # print("after model update:")
            # print("y_prob:",y_prob)
            # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])
            recourseFailCnt = len(y_prob[y_prob < 0.5])
            print("recourseFailCnt",recourseFailCnt)
            recourseFailRate = recourseFailCnt / len(x[recourseIndex])
            print("recourseFailRate : ",recourseFailRate)
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
        modelParams = list(model.parameters())
        modelParameter = np.concatenate((weights,bias))
        resultParameter = np.concatenate((modelParams[0].data.reshape(-1),modelParams[1].data))
        # print("Before update: ")
        # print(weights,bias)
        # print("After update: ")
        # print(modelParams[0].data.reshape(-1),modelParams[1].data)

        parameterL2 = np.linalg.norm(resultParameter - modelParameter)

        self.PDt.append(parameterL2)

# weight = pt.from_numpy(np.random.gamma(3,1,20))
# print(train.x)
# print(train.y)
BinomialProb = 0.01
ex1 = Example1(model, pca, train, test, sample)
ex1.save_directory = DIRECTORY

ani1 = ex1.animate_all(80)        
# ani1 = ex1.animate_all(200)
ani1.save(os.path.join(DIRECTORY, "ex1.mp4"))

# ex1.draw_PDt()
ex1.draw_PDt()
ex1.draw_EFT(80)
ex1.draw_R20_EFT(80,10)
ex1.draw_R20_EFT(80,20)
ex1.draw_R20_EFT(80,58)

# ex1.draw_EFT(200)
# ex1.draw_R20_EFT(200,40)
# ex1.draw_R20_EFT(200,60)

ex1.draw_Fail_to_Recourse()
display(ex1.EFTdataframe)
ex1.plot_matricsA()
