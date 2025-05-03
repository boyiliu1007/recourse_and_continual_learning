import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


class Recourse(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.action = nn.Parameter(pt.zeros(size))  
        

    def forward(self, x: pt.Tensor, weight: pt.Tensor = None):
        a = self.action
        x = x + a
        return_act = a.clone().detach()
        return x, return_act
    
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    optimizer = optim.SGD(r_model.parameters(), lr=0.5)
    pt.manual_seed(42)
    cost_constant = pt.tensor([2, 2])
    # normalize the cost constant and make it sum to 1
    cost_constant = cost_constant / cost_constant.sum()

    criterion = nn.BCELoss()
    r_model.train()

    for epoch in range(max_epochs):
        x_hat, return_act = r_model(dataset.x)
        y_hat = c_model(x_hat)
        
        # relu loss version
        # output_margin = y_hat - 0.7
        # target_margin = pt.ones_like(y_hat) * 0.001  # push slightly over
        # penalty_weight = 10
        # margin_loss = (pt.relu(target_margin - output_margin)* penalty_weight).mean()
        
        # bce loss version
        target = pt.ones_like(y_hat)
        target *= 1
        bce_loss = criterion(y_hat, target)
        
        # action cost (here use squared error)
        action_cost = ((x_hat - dataset.x) ** 2).mean(0)
        weighted_action_cost = (cost_constant * action_cost).sum()
        
        # final loss function
        loss = bce_loss + 0.5 * weighted_action_cost

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch == max_epochs - 1):
            print(f"Recourse Loss: {loss.item():.4f}")

    # print(r_model.action.grad)
    dataset.x = x_hat.detach() 
    dataset.y = (c_model(dataset.x) > 0.5).float()

    return dataset, return_act.detach()

    