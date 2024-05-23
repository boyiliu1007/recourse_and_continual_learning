import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
# def training(model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list | None = None):
def training(model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None, lr = 0.1):
    print("lr: ", lr)
    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model.train()

    for _ in range(max_epochs):
        outputs = model(dataset.x)
        optimizer.zero_grad()
        loss = criterion(outputs, dataset.y)
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()
        scheduler.step(loss)
        if loss_list is not None:
            loss_list.append(loss.item())
    model.eval()