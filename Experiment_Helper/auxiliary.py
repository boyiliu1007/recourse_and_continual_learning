import torch as pt
import numpy as np
import math
import pandas as pd
import datetime

def getFunc(x):
    # Function definition
    return math.log(x - 0.9) + 2.5
    
def getWeights(feature_num, type = 'uniform'):
    # type can be chosen from uniform, log
    if(type == 'uniform'):
        return pt.from_numpy(np.ones(feature_num)) / feature_num
    elif(type == 'log'):
        weights = np.array([getFunc(i) for i in range(1, feature_num + 1)])
        weights = weights / np.sum(weights)  # Normalize the weights
        return pt.from_numpy(weights)
    else:
        print("type is incorrect at getWeights")

def update_train_data(train, sample, model, type = 'all', expected_size = None):
    #type can be chosen from none, all, mixed
    if type == 'none':
        return train, pt.empty(0, dtype=pt.bool)


    if (expected_size is not None) and (expected_size > train.x.shape[0]):
        size = expected_size
    else:
        size = train.x.shape[0]

    sample_indices = pt.randperm(sample.x.shape[0])[:size]
    # print(f"sample_indices: {sample_indices}")
    sampled_x = sample.x[sample_indices]
    # print(f"sampled_x: {sampled_x}")

    if type == 'mixed':
        half_size = size // 2

        retain_indices = pt.randperm(train.x.shape[0])[:half_size]
        modify_indices = pt.tensor(np.setdiff1d(np.arange(size), retain_indices.numpy()))

        new_x = pt.zeros((size, *train.x.shape[1:]), dtype=train.x.dtype)
        new_y = pt.zeros(size, dtype=train.y.dtype)

        new_x[modify_indices] = sampled_x[:modify_indices.shape[0]]
        new_x[retain_indices] = train.x[retain_indices]
        new_y[retain_indices] = train.y[retain_indices]
        with pt.no_grad():
            y_prob: pt.Tensor = model(new_x[modify_indices])
        y_prob = y_prob.squeeze(1)
        new_y[modify_indices] = pt.where(y_prob > 0.5, 1.0, 0.0)
        isNew = pt.zeros(size, dtype=pt.bool)
        isNew[modify_indices] = True
        
        train.x = new_x
        train.y = new_y
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

class FileSaver:
    def __init__(self, fail_to_recourse, overall_acc_list, jsd_list, avgRecourseCost, avgNewRecourseCost, avgOriginalRecourseCost, t_rate_list, model_shift_list):
        self.failToRecourse = fail_to_recourse
        self.overall_acc_list = overall_acc_list
        self.jsd_list = jsd_list
        self.avgRecourseCost = avgRecourseCost
        self.avgNewRecourseCost = avgNewRecourseCost
        self.avgOriginalRecourseCost = avgOriginalRecourseCost
        self.t_rate_list = t_rate_list
        self.model_shift_list = model_shift_list

    def save_to_csv(self, recourse_num, threshold, acceptance_rate, cost_weight, dataset, current_time, directory = ''):
        filename = f"{recourse_num}_{threshold}_{acceptance_rate}_{cost_weight}_{dataset}_{current_time}.csv"
        if directory:
            directory = directory.rstrip('/') + '/'
            filename = directory + filename

        print(len(self.failToRecourse))
        print(len(self.overall_acc_list))
        print(len(self.jsd_list))
        print(len(self.avgRecourseCost))
        print(len(self.avgNewRecourseCost))
        print(len(self.avgOriginalRecourseCost))
        print(len(self.t_rate_list))
        print(len(self.model_shift_list))

        # since short term accuracy cannot calculate the first 2 element so insert two 0s here
        self.overall_acc_list.insert(0, 0)
        self.overall_acc_list.insert(0, 0)
        self.overall_acc_list.insert(0, 0)
        
        data = {
            'failToRecourse': self.failToRecourse,
            'acc': self.overall_acc_list,
            'jsd': self.jsd_list,
            'avgRecourseCost': self.avgRecourseCost,
            'avgNewRecourseCost': self.avgNewRecourseCost,
            'avgOriginalRecourseCost': self.avgOriginalRecourseCost,
            't_rate': self.t_rate_list,
            'model_shift': self.model_shift_list
        }
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"File saved as: {filename}")