import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze(1)
    
def training(model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None, lr=0.4):
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    criterion = nn.MSELoss(reduction='sum') 
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:

            outputs = model(inputs)
            loss = 0.5 * criterion(outputs, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_list is not None:
                loss_list.append(loss.item())

        if(epoch == max_epochs - 1):
            print(f"Model Loss: {loss.item():.4f}")
    
    return model

def training_update_bias_only(model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list = None, lr=0.4):
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    criterion = nn.MSELoss(reduction='sum') 
    
    # Select only bias parameters
    bias_params = [param for name, param in model.named_parameters() if 'bias' in name]
    optimizer = optim.SGD(bias_params, lr=lr)

    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = 0.5 * criterion(outputs, targets)  # squared error loss divided by 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_list is not None:
                loss_list.append(loss.item())

        if epoch == max_epochs - 1:
            print(f"Model Loss: {loss.item():.4f}")
    
    return model