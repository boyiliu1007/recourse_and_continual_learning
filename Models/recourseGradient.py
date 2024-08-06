import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt

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
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None,threshold = 1.0):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(r_model.parameters(), 0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # threshold = pt.ones(dataset.y.size())
    threshold = pt.ones(dataset.y.size()).fill_(threshold)
    print("threshold : ",threshold)

    r_model.train()
    for _ in range(max_epochs):
        # optimizer.zero_grad()
        x_hat,cost = r_model(dataset.x)
        y_hat = c_model(x_hat)
        #lamda = 0.5
        # loss = criterion(y_hat, dataset.y) + 0.3 * pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
        loss = criterion(y_hat, threshold) + 0.3 * pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
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

    with pt.no_grad():
        dataset.x,cost = r_model(dataset.x)
         # x_hat,cost = r_model(dataset.x)
        if cost_list is not None:
            L2_cost = pt.pow(pt.sum((cost * weight) * (cost * weight)),1/2)
            cost_list.append(L2_cost.item())
            print("Recourse cost: ",L2_cost.item())
            
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