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
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None,threshold = 1.0,q3RecourseCost: list = None,recourseModelLossList: list = None):
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
    recourseModelLossList.append(loss.item())

    with pt.no_grad():
        recourseCostLimit = 100
        recourseLambda = 0.6
        # dataset.x,cost = r_model(dataset.x)
        # print("Before recourse:",dataset.x)
        recourseX,cost = r_model(dataset.x)
        # print("After Recourse",dataset.x)
        # print("recourseX: ",recourseX)
        # dataset.x = 
        # for name, param in r_model.named_parameters():
        #     print(name, param.data)
        # print("cost: ",cost)
         # x_hat,cost = r_model(dataset.x)
        if cost_list is not None:
            avgRecourseCost = 0.0
            recourseCostList = []
            # print("cost:",cost)
            for idx,t in enumerate(cost):
                # print("cost: ",t)
                L2_cost = pt.pow(pt.sum((t * weight) * (t * weight)),1/2)
                # print("Recourse cost: ",L2_cost.item())
                #Gradient = f(x)' - lambda * cost(x,x') > 0
                a = c_model(recourseX[idx])
                # print("f(x)':",a)
                recourseGradient = a -  (1 / recourseLambda) * pt.pow(pt.sum((t * weight) * (t * weight)),1/2)
                # print("recourse gradient: ",recourseGradient)
                #set the recourse cost limit
                # if L2_cost < recourseCostLimit:
                if recourseGradient >= 0:
                    # print("pass the limit!")
                    dataset.x[idx] = recourseX[idx]
                avgRecourseCost += L2_cost
                recourseCostList.append(L2_cost.item())
                # cost_list.append(L2_cost.item())
                # print("Recourse cost: ",L2_cost.item())
            # print("After Recourse",dataset.x)
            avgRecourseCost /= len(cost)
            if q3RecourseCost is not None:
                # print("recourseCostList :",recourseCostList)
                # print("q3: ",np.quantile(recourseCostList,0.75))
                q3RecourseCost.append(np.quantile(recourseCostList,0.75))
            
            
            cost_list.append(avgRecourseCost.item())
            print("avgRecourseCost cost: ",avgRecourseCost.item())
            
        #Recourse cost limit set as 0.35
        # if L2_cost.item() < 0.3:
        #     print("pass the cost limit")
        #     dataset.x = x_hat
    # draw_statistic(loss_list,mode='loss')
    
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Round')
    plt.ylabel('loss')
    plt.title('loss')
    plt.savefig('GradientRecourse_loss_init.png')