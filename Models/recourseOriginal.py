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

    # def forward(self, x: pt.Tensor, weight: pt.Tensor | None = None):
    def forward(self, x: pt.Tensor, weight: pt.Tensor = None):
        a = self.action
        # print("a",a)

        #to be implement
        #add cost funtion weight

        # if weight is not None:
        #     a = a * weight
        x = x + a
        cost = deepcopy(a)
        return x,cost


# def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor | None = None, loss_list: list | None = None):
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None,cost_list = None,threshold = 1.0):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(r_model.parameters(), 0.1)
    threshold = pt.ones(dataset.y.size()).fill_(threshold)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # print("threshold : ",threshold)


    r_model.train()
    for _ in range(max_epochs):
        for name, param in r_model.named_parameters():
            print(name, param.data)
        x_hat,cost = r_model(dataset.x)
        # print("c_model's param: ",c_model.parameters())
        # with pt.no_grad():
        y_hat = c_model(x_hat)
        # loss = criterion(y_hat, dataset.y)
        loss = criterion(y_hat, threshold)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()
        c_model.zero_grad()
        # scheduler.step(loss)
        if loss_list is not None:
            loss_list.append(loss.item())
    r_model.eval()

    with pt.no_grad():
        dataset.x,cost = r_model(dataset.x)
        for idx,t in enumerate(cost):
            a = c_model(dataset.x[idx])
            print("f(x)':",a)
    # draw_statistic(loss_list,mode='loss')