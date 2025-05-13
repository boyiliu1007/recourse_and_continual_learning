import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

#Test weight Recourse

class Recourse(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.action = nn.Parameter(pt.zeros(size))  

    def forward(self, x: pt.Tensor, weight: pt.Tensor = None):
        a = self.action
        x = x + a
        return_act = a.detach().clone()
        return x, return_act


# def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor | None = None, loss_list: list | None = None):
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None,threshold = 1.0,q3RecourseCost: list = None,recourseModelLossList: list = None, isNew = None, new_cost_list = None, original_cost_list = None):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(r_model.parameters(), lr=0.5)
    
    # threshold = pt.ones(dataset.y.size())
    # threshold_v = pt.ones(dataset.y.size()).fill_(threshold)
    

    r_model.train()
    for epoch in range(max_epochs):
        x_hat, return_act = r_model(dataset.x)
        y_hat = c_model(x_hat)
    
        # bceloss = criterion(y_hat, threshold_v)
        
        # relu loss version
        # output_margin = y_hat - threshold
        # target_margin = pt.ones_like(y_hat) * 0.001  # push slightly over
        # penalty_weight = 10
        # margin_loss = (pt.relu(target_margin - output_margin)* penalty_weight).mean()
        
        # bce loss version
        target = pt.ones_like(y_hat)          
        target *= threshold
        bce_loss = criterion(y_hat, target)
        weight = weight / weight.sum()
        
        # action cost (here use squared error)
        action_cost = (x_hat - dataset.x) ** 2
        weighted_action_cost = (weight * action_cost).sum()

        
        # final loss function
        loss = bce_loss + 0.001 * weighted_action_cost

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch == max_epochs - 1):
            print(f"Recourse Loss: {loss.item():.4f}")

    # print(r_model.action.grad)
    dataset.x = x_hat.detach() 
    dataset.y = (c_model(dataset.x) > 0.5).float()
    score = c_model(dataset.x)
    print("Average Recourse Score:", score.mean().item())

    if cost_list is not None:
        avgRecourseCost = 0.0
        avgOriginalRecourseCost = 0.0
        avgNewRecourseCost = 0.0
        newCount = 0
        with pt.no_grad():
            recoursedX,action = r_model(dataset.x)
        sqr_action = action ** 2
        weighted_action_cost = (weight * sqr_action).sum(dim = 1)
        avgRecourseCost = weighted_action_cost.mean().item()
        for idx,t in enumerate(weighted_action_cost):
            if idx < isNew.size(0) and isNew[idx]:
                newCount += 1
                avgNewRecourseCost += t.item()
            else:
                avgOriginalRecourseCost += t.item()
        
        if newCount == 0:
            avgNewRecourseCost = 0.0
        else:
            avgNewRecourseCost /= newCount
        avgOriginalRecourseCost /= len(weighted_action_cost) - newCount
        cost_list.append(avgRecourseCost)
        new_cost_list.append(avgNewRecourseCost)
        original_cost_list.append(avgOriginalRecourseCost)
    return dataset, return_act.detach()