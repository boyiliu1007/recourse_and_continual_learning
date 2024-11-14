import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor | None = None, loss_list: list | None = None):
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None,cost_list = None,threshold = 1.0,recourseModelLossList: list = None):
    loss: pt.Tensor
    print("Enter Recourse : ",c_model.state_dict())
    # a = pt.tensor(dataset.x,requires_grad=True,dtype=pt.float32)
    a = dataset.x.clone().detach().requires_grad_(True)
    x = deepcopy(a)
    criterion = nn.HuberLoss()
    
    #無法使用Binary Cross Entropy，因為我們是使用Threshold來計算Loss，不是使用True label
    # criterion = nn.BCELoss()
    optimizer = optim.Adam([x], 0.1)
    threshold = pt.ones(dataset.y.size()).fill_(threshold)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # print("threshold : ",threshold)
    # print(f"x.tolist : {[[val for val in point]for point in x.tolist()]}")
    # print(f"Before Recourse x : {[[round(val,3) for val in points] for points in x.tolist()]}",)
    for step in range(max_epochs):
        # for name, param in r_model.named_parameters():
        #     print(name, param.data)
        optimizer.zero_grad()
        # print("x_hat : ",x_hat)
        # print("c_model's param: ",c_model.parameters())
        # with pt.no_grad():
        y_hat = c_model(x)
        # loss = criterion(y_hat, dataset.y)
        loss = criterion(y_hat, threshold)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        c_model.zero_grad()
        # scheduler.step(loss)
        if step % 60 == 0:
            # print(f'Step {step}, Loss: {loss.item():.3f}, Updated x: {[[round(val,3)for val in points] for points in x.tolist()]}')
            print(f'Step {step}, Loss: {loss.item():.3f}')
        if loss_list is not None:
            loss_list.append(loss.item())
    recourseModelLossList.append(loss.item())
    result = x - a
    dataset.x = x.detach().numpy()
    print(f"cost : {[[round(val,3) for val in points] for points in result.tolist()]}")
    # print(f"after Recourse x : {[[round(val,3)for val in points] for points in x.tolist()]}",)

    # print("Before Recourse dataset.x : ",dataset.x)
    # with pt.no_grad():
    #     dataset.x,cost = r_model(dataset.x)
    #     # print("cost : ",cost)
    #     # print("After Recourse dataset.x : ",dataset.x)
    #     for idx,t in enumerate(cost):
    #         a = c_model(dataset.x[idx])
            # print("f(x)':",a)
    # draw_statistic(loss_list,mode='loss')
    
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Round')
    plt.ylabel('loss')
    plt.title('loss')
    plt.savefig('OriginalRecourse_loss.png')
    plt.close()