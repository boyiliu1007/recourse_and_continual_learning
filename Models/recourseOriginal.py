import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset

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
        return x


# def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor | None = None, loss_list: list | None = None):
def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None,threshold = 1.0):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(r_model.parameters(), 0.1)
    threshold = pt.ones(dataset.y.size()).fill_(threshold)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    print("threshold : ",threshold)

    r_model.train()
    for _ in range(max_epochs):
        # optimizer.zero_grad()
        x_hat = r_model(dataset.x)
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
        dataset.x = r_model(dataset.x)
    # draw_statistic(loss_list,mode='loss')