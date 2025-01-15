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

def update_train_data(train, sample, model, type = 'all'):
    #type can be chosen from none, all, mixed
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

class FileSaver:
    def __init__(self, fail_to_recourse, overall_acc_list, jsd_list, avgRecourseCost, avgNewRecourseCost, avgOriginalRecourseCost):
        self.failToRecourse = fail_to_recourse
        self.overall_acc_list = overall_acc_list
        self.jsd_list = jsd_list
        self.avgRecourseCost = avgRecourseCost
        self.avgNewRecourseCost = avgNewRecourseCost
        self.avgOriginalRecourseCost = avgOriginalRecourseCost

    def save_to_csv(self, recourse_num, threshold, acceptance_rate, cost_weight, dataset, directory = ''):
        current_time = datetime.datetime.now().strftime("%d-%H-%M")
        filename = f"{recourse_num}_{threshold}_{acceptance_rate}_{cost_weight}_{dataset}_{current_time}.csv"
        if directory:
            directory = directory.rstrip('/') + '/'
            filename = directory + filename

        # since short term accuracy cannot calculate the first 2 element so insert two 0s here
        self.overall_acc_list.insert(0, 0)
        self.overall_acc_list.insert(0, 0)

        data = {
            'failToRecourse': self.failToRecourse,
            'acc': self.overall_acc_list,
            'jsd': self.jsd_list,
            'avgRecourseCost': self.avgRecourseCost,
            'avgNewRecourseCost': self.avgNewRecourseCost,
            'avgOriginalRecourseCost': self.avgOriginalRecourseCost
        }
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"File saved as: {filename}")