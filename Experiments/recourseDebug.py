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
# from Models.MLP import MLP, training
# from Models.recourseGradient import recourse
from Models.recourseOriginal import recourse
# from Models.recourseTest import recourse
from Config.config import train, test, sample, model
# from Config.MLP_config import MLP_model
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

#binomial Recourse distribution    Prob = 0.7
class Example1(Helper):
    '''
    Update Method Steps:
    1. Selects a random subset of `sample` with a approximately size of `train.size * 0.02`.
    2. Performs recourse on the selected subset of 'sample' with binomial distribution and preserve their original labels.
    3. Replaces a corresponding part of the training set with the updated samples.
    4. Refits the model with the modified training data.
    '''

    #add first k              k maybe 40% 80% 100%

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        print("round: ",self.round)
        # self.model = LogisticRegression(train.x.shape[1],1)
        # training(self.model, self.train, 150,self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)
        # print("train.x.shape: ",train.x.shape[0])
        #get model parameters before model updated
        modelParams = list(self.model.parameters())
        weights = deepcopy(modelParams[0].data.reshape(-1))
        bias = deepcopy(modelParams[1].data)
        
        # print("Before update: ")
        # print(modelParams)
        # print(weights,bias)

        # substitute 1 / 4 train data
        size = train.x.shape[0] // 10
        
        #substitute all train dada
        # size = train.x.shape[0]
        print(f"size : {size}")
        
        i = np.random.choice(sample.x.shape[0], size, False)
        x = sample.x[i]
        # print("x:",x)

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x)

        # print("predict: ",y_prob.data)
        y_pred = y_prob.flatten() < 0.5
        # print("predict: ",y_prob.data[y_pred])

        # Performs recourse on the selected samples with binomial distributed destination.
        binomialData = np.random.binomial(1,BinomialProb,np.count_nonzero(y_pred)).astype(bool)
        # print("binomial data: ",binomialData)
        recourseIndex = deepcopy(y_pred)
        # print("before binomial",recourseIndex)
        recourseIndex[recourseIndex == True] = pt.from_numpy(binomialData)
        # y_pred[y_pred == True] = pt.from_numpy(binomialData)
        # print("y_pred",y_pred)
        
        # print("binomial y_pred",recourseIndex)
        # print("x[recourseIndex] : ",x[recourseIndex])
#           binomial y_pred tensor([False, False, False, False, False, False, False, False, False, False,
#           False, False, False, False, False, False, False, False, False, False,
#           False, False, False, False, False])
#           x[recourseIndex] :  tensor([], size=(0, 20))
        sub_sample = Dataset(x[recourseIndex], pt.full((recourseIndex.count_nonzero(), 1),1.0))
        # print("sub_sample : ",sub_sample)
        # print("sub_sample x: ",sub_sample.x)
        # print("sub_sample y: ",sub_sample.y)
        # print("len(sub_sample) : ",len(sub_sample))
        print("Not enter Recourse",self.model.state_dict())

        # recourse(model, sub_sample, 10,weight,loss_list=[])
        if len(sub_sample) > 0:
            print("Real recourse")
            # print("sub_sample: ",sub_sample.y)
            # print("sub_sample.y size: ",sub_sample.y.size())
            # recourse(model, sub_sample, 200,weight,loss_list=[],threshold=0.9,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
            recourse(self.model,sub_sample,200,loss_list=[],threshold=0.9,recourseModelLossList=self.recourseModelLossList)
            # test = deepcopy(x)

            #change the attributes that performed recourse
            # x[recourseIndex] = sub_sample.x
            x[recourseIndex] = pt.tensor(sub_sample.x)
            # print("x[recourseIndex].requires_grad: ",x[recourseIndex].requires_grad)
            # print("x[recourseIndex].grad: ",x[recourseIndex].grad)

            # print("after recourse")
            # print("check eq :",pt.eq(test,x))
            # with pt.no_grad():
            #     y_prob: pt.Tensor = model(sub_sample.x)
            #     print("y_prob: ",y_prob)
            # print("after recourse predict: ",y_prob.data[y_pred])
            
            #計算Recourse失敗率
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(x[recourseIndex])

            # print("x[y_pred] : ",x[y_pred])
            # print("after model update:")
            # print("y_prob:",y_prob)
            # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])
            recourseFailCntBeforeUpdate = len(y_prob[y_prob < 0.5])
            # print("recourseFailCnt",recourseFailCnt)
            recourseFailRateBeforeUpdate = recourseFailCntBeforeUpdate / len(x[recourseIndex])
            # print("recourseFailRate : ",recourseFailRate)
            self.failToRecourseBeforeModelUpdate.append(recourseFailRateBeforeUpdate)
        else: #對sample data中 label = 0的data points做Recourse
            recoursePoints = Dataset(x[y_pred], pt.ones((y_pred.count_nonzero(), 1)))
            # print("Before Recourse recoursePoints.x : ",recoursePoints.x)
            a = deepcopy(recoursePoints.x)
            # recourse(model, recoursePoints, 200,weight,loss_list=[],threshold=0.9,cost_list=self.avgRecourseCost_list,q3RecourseCost=self.q3RecourseCost,recourseModelLossList=self.recourseModelLossList)
            recourse(self.model,recoursePoints,200,loss_list=[],threshold=0.9,recourseModelLossList=self.recourseModelLossList)
            
            # print("len(recoursePoints) : ",len(recoursePoints))
            # print("Before model update recourse:")
            # with pt.no_grad():
            #     y_prob: pt.Tensor = model(recoursePoints.x)
            # print("y_prob: ",y_prob)
            # recourseAcc = len(y_prob[y_prob >= 0.5])
            # recourseAccRate = recourseAcc / len(recoursePoints)
            # print("recourseAccRate:",recourseAccRate)
            
            #計算model update之前的Recourse失敗率
            print("Imagine Recourse:")
            # print("recoursePoints.x : ",recoursePoints.x)
            # self.failToRecourse.append(0)
            result = pt.tensor(recoursePoints.x)
            result = result - a
            # print(f"After Recourse recoursePoints.x : {[[round(val,3) for val in points] for points in result.tolist()]}",)
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(result)
                # print("y_prob : ",y_prob)
            
            recourseFailCntBeforeUpdate = len(y_prob[y_prob < 0.5])
            # print("recourseFailCnt",recourseFailCnt)
            recourseFailRateBeforeUpdate = recourseFailCntBeforeUpdate / len(recoursePoints)
            # print("recourseFailRate : ",recourseFailRate)
            self.failToRecourseBeforeModelUpdate.append(recourseFailRateBeforeUpdate)
            

        j = np.random.choice(train.x.shape[0], size, False)
        # print("len(j): ",len(j))
        # print("j : ",j)

        # #pseudou-labeling(based on j-1's model and recourse)
        # with pt.no_grad():
        #     y_prob: pt.Tensor = model(x)

        # y_pred = y_prob.flatten() < 0.5
        self.train.x[j] = x
        # train.y[j, 0] = (~y_pred).float()
        # print("self.train.y shape: ",self.train.y.shape)
        # print("y_pred shape: ",y_pred.shape)
        # self.train.y[j, 0] = (~y_pred).float()
        self.train.y[j] = (~y_pred).float()

        val_data = Dataset(self.train.x[j], self.train.y[j])
        self.validation_list.append(val_data)
        # sample_model = LogisticRegression(val_data.x.shape[1], 1)
        sample_model = LogisticRegression(val_data.x.shape[1], 1)
        sample_model.train()
        training(sample_model, val_data, 30,self.test)
        self.Aj_tide_list.append(self.calculate_accuracy(sample_model(val_data.x), val_data.y))

        #紀錄新增進來的sample資料
        self.addEFTDataFrame(j)

        # self.model = LogisticRegression(train.x.shape[1],1)
        training(self.model, self.train, 100,self.test,loss_list=self.RegreesionModelLossList,val_loss_list=self.RegreesionModel_valLossList,printLoss=True)
        print("Initial and train model : ",self.model.state_dict())
        
         #Calculate the proportion of test data labels that remain consistent in the new round of labeling.   
        with pt.no_grad():
          y_test: pt.Tensor = self.model(self.test.x)
        test_trueLabelIdx = self.test.y.flatten() > 0.5
        #在test data中label為1的index的點
        y_test_trueLabel = y_test[test_trueLabelIdx]
        #calculate the proportion
        testDataConsistentRatio = pt.count_nonzero(y_test_trueLabel > 0.5).item() / len(test.y[test_trueLabelIdx])
        self.fairRatio.append(testDataConsistentRatio)
        # print(y_test_trueLabel > 0.5)
        # print("testDataConsistentRatio : ",testDataConsistentRatio)

        # calculate the overall accuracy
        self.overall_acc_list.append(self.calculate_AA(self.model, self.validation_list))
        # evaluate memory stability
        self.memory_stability_list.append(self.calculate_BWT(self.model, self.validation_list, self.Ajj_performance_list))
        # evaluate memory plasticity
        self.memory_plasticity_list.append(self.calculate_FWT(self.Ajj_performance_list, self.Aj_tide_list))


        # #紀錄Fail_to_Recourse
        if len(sub_sample) > 0:
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(x[recourseIndex])

            # print("x[y_pred] : ",x[y_pred])
            # print("after model update:")
            # print("y_prob:",y_prob)
            # print("y_prob[y_prob < 0.5]",y_prob[y_prob < 0.5])
            recourseFailCnt = len(y_prob[y_prob < 0.5])
            # print("recourseFailCnt",recourseFailCnt)
            recourseFailRate = recourseFailCnt / len(x[recourseIndex])
            # print("recourseFailRate : ",recourseFailRate)
            self.failToRecourse.append(recourseFailRate)
        else:
            # print("no Recourse:")
            # print("Imagine Recourse:")
            # print("recoursePoints.x : ",recoursePoints.x)
            # self.failToRecourse.append(0)
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(pt.tensor(recoursePoints.x))
                # print("after model update:")
                # print("y_prob : ",y_prob)
            
            recourseFailCnt = len(y_prob[y_prob < 0.5])
            # print("recourseFailCnt",recourseFailCnt)
            recourseFailRate = recourseFailCnt / len(recoursePoints)
            # print("recourseFailRate : ",recourseFailRate)
            self.failToRecourse.append(recourseFailRate)

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

weight = pt.from_numpy(np.ones(train.x.shape[1])) / train.x.shape[1]
print("outside: ",weight)
# cost = []
# print(train.x)
# print(train.y)
BinomialProb = 0.01
ex1 = Example1(model, pca, train, test, sample)
# ex1 = Example1(MLP_model, pca, train, test, sample)
ex1.save_directory = DIRECTORY
# ani1 = ex1.animate_all(80)

# ani1 = ex1.animate_all(80)
ani1 = ex1.animate_all(80)
ani1.save(os.path.join(DIRECTORY, "ex1.mp4"))

# print("cost len:",len(cost))
# if len(cost) > 0:
#     avgRecourseCost = sum(cost) / len(cost)
#     print("avgRecourseCost:",avgRecourseCost)

# ex1.draw_PDt()
ex1.draw_PDt()
ex1.draw_EFT(80)
ex1.draw_R20_EFT(80,10)
ex1.draw_R20_EFT(80,20)
# ex1.draw_avgRecourseCost()
ex1.draw_testDataFairRatio()
# ex1.draw_q3RecourseCost()
# ex1.draw_failToRecourseCompareToNormalModel()
# ex1.draw_avgRecourseCostCompareToNormalModel()

# ex1.draw_EFT(150)
# ex1.draw_R20_EFT(150,30)
# ex1.draw_R20_EFT(150,60)

# ex1.draw_EFT(240)
# ex1.draw_R20_EFT(240,23)
# ex1.draw_R20_EFT(240,40)
# ex1.draw_R20_EFT(240,58)
ex1.draw_Fail_to_Recourse()
ex1.draw_failToRecourseBeforeModelUpdate()
ex1.draw_recourseModelLoss()
ex1.draw_RegressionModelLoss()
ex1.draw_RegressionModelValLoss()
ex1.draw_Fail_to_Recourse_with_Model_Loss()
display(ex1.EFTdataframe)
ex1.plot_matricsA()
ex1.plot_Ajj()