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
        # self.mask = pt.zeros(size)  

        # self.mask[:, :17] = 1  

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(r_model.parameters(), lr=0.1)
    
    # threshold = pt.ones(dataset.y.size())
    threshold_v = pt.ones(dataset.y.size()).fill_(threshold)
    

    r_model.train()
    for _ in range(max_epochs):
        
        x_hat,cost = r_model(dataset.x)
        
        y_hat = c_model(x_hat)
        # loss = criterion(y_hat, dataset.y) + 0.3 * pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
        # bceloss = criterion(y_hat, threshold_v)

        output_margin = y_hat - 0.7
        target_margin = pt.ones_like(y_hat) * 0.001  # push slightly over
        
        margin_loss = (pt.relu(target_margin - output_margin)* weight * 100000).mean()

        cost_constraint = pt.pow(pt.sum(weight * cost * cost), 1/2)
        loss = margin_loss + 1 * cost_constraint
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss_list is not None:
            loss_list.append(loss.item())

    dataset.x = x_hat.detach().clone()
    dataset.y = (y_hat.detach().clone() > 0.5).float()
    r_model.eval()
    if(recourseModelLossList is not None):
        recourseModelLossList.append(loss.item())
    
    with pt.no_grad():
        y_hat = c_model(dataset.x)
        print("y_hat",y_hat.squeeze())
        print("recourse action", cost)

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
                L2_cost = pt.pow(pt.sum(weight * t * t), 1/2)
                if (idx < isNew.size(0) and isNew[idx]):
                    newCount += 1
                    avgNewRecourseCost += L2_cost
                else:
                    avgOriginalRecourseCost += L2_cost
                a = c_model(recourseX[idx])
                recourseGradient = a -  (1 / recourseLambda) * pt.pow(pt.sum(weight * t * t), 1/2)
                if recourseGradient >= 0:
                    dataset.x[idx] = recourseX[idx]
                avgRecourseCost += L2_cost
                recourseCostList.append(L2_cost.item())

            if len(cost) == 0:
                avgRecourseCost = -1
            else:
                avgRecourseCost /= len(cost)
                if pt.any(isNew != 0):
                    avgNewRecourseCost /= newCount
                    avgOriginalRecourseCost /= (len(cost) - newCount)
                else:
                    avgNewRecourseCost = pt.zeros(1)
                    avgOriginalRecourseCost = pt.zeros(1)

                
            if q3RecourseCost is not None:
                q3RecourseCost.append(np.quantile(recourseCostList,0.75))

            
            cost_list.append(avgRecourseCost.item())
            original_cost_list.append(avgOriginalRecourseCost.item())
            new_cost_list.append(avgNewRecourseCost.item())
            # print("avgRecourseCost cost: ",avgRecourseCost.item())
            # print("avgNewRecourseCost: ",avgNewRecourseCost.item(), newCount)
            # print("avgOriginalRecourseCost ", avgOriginalRecourseCost.item(), len(cost) - newCount)

    
    return dataset