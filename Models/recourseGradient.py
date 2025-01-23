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
        self.mask = pt.zeros(size)  

        self.mask[:, :17] = 1  

    def forward(self, x: pt.Tensor, weight: pt.Tensor = None):
        # a = self.action * self.mask.detach()
        a = self.action
        x = x + a
        cost = a.detach().clone()
        return x, cost


# def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor | None = None, loss_list: list | None = None):
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None,threshold = 1.0,q3RecourseCost: list = None,recourseModelLossList: list = None, isNew = None, new_cost_list = None, original_cost_list = None):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(r_model.parameters(), 0.1)
    # print("Enter Recourse",c_model.state_dict())
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # threshold = pt.ones(dataset.y.size())
    threshold = pt.ones(dataset.y.size()).fill_(threshold)
    # print("threshold : ",threshold)

    r_model.train()
    for _ in range(max_epochs):
        # optimizer.zero_grad()
        x_hat,cost = r_model(dataset.x)
        # print("cost: ",cost)
        y_hat = c_model(x_hat)
        # print("y_hat: ",y_hat)
        #lamda = 0.5
        # loss = criterion(y_hat, dataset.y) + 0.3 * pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
        loss = criterion(y_hat, threshold) + 0.3 * pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
        # print("loss: ",loss.item())
        #clear gradient
        optimizer.zero_grad()
        #calculate gradient
        loss.backward()
        #do gradient descent
        optimizer.step()
        # optimizer.zero_grad()
        #why need c_model.zero_grad()?
        c_model.zero_grad()
        # scheduler.step(loss)
        if loss_list is not None:
            loss_list.append(loss.item())
    r_model.eval()
    if(recourseModelLossList is not None):
        recourseModelLossList.append(loss.item())

    with pt.no_grad():
        recourseCostLimit = 100
        recourseLambda = 0.6
        recourseX,cost = r_model(dataset.x)
        if cost_list is not None:
            avgRecourseCost = 0.0
            avgOriginalRecourseCost = 0.0
            avgNewRecourseCost = 0.0
            newCount = 0
            recourseCostList = []
            
            for idx,t in enumerate(cost):
                # if the idx matches isNew then count it as new recoursecost
                # otherwise not
                L2_cost = pt.pow(pt.sum((t * weight) * (t * weight)),1/2)
                if(isNew[idx]):
                    newCount += 1
                    avgNewRecourseCost += L2_cost
                else:
                    avgOriginalRecourseCost += L2_cost
                a = c_model(recourseX[idx])
                recourseGradient = a -  (1 / recourseLambda) * pt.pow(pt.sum((t * weight) * (t * weight)),1/2)
                if recourseGradient >= 0:
                    dataset.x[idx] = recourseX[idx]
                avgRecourseCost += L2_cost
                recourseCostList.append(L2_cost.item())
            if len(cost) == 0:
                avgRecourseCost = -1
            else:
                avgRecourseCost /= len(cost)
                avgNewRecourseCost /= newCount
                avgOriginalRecourseCost /= (len(cost) - newCount)
                
            if q3RecourseCost is not None:
                q3RecourseCost.append(np.quantile(recourseCostList,0.75))

            
            cost_list.append(avgRecourseCost.item())
            original_cost_list.append(avgOriginalRecourseCost.item())
            new_cost_list.append(avgNewRecourseCost.item())
            print("avgRecourseCost cost: ",avgRecourseCost.item())
            print("avgNewRecourseCost: ",avgNewRecourseCost.item(), newCount)
            print("avgOriginalRecourseCost ", avgOriginalRecourseCost.item(), len(cost) - newCount)