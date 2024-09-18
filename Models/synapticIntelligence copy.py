import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset

# SI class to manage parameter importance
class SynapticIntelligence:
    def __init__(self, model):
        self.model = model
        self.prev_params = {}
        self.omega = {}
        self.param_updates = {}
        self.epsilon = 0.1  # Small value to avoid division by zero

        for name, param in model.named_parameters():
            self.prev_params[name] = param.data.clone()
            self.omega[name] = pt.zeros_like(param)
            self.param_updates[name] = pt.zeros_like(param)

    def update_omega(self, train: Dataset, criterion):
        self.model.eval()
        for inputs, labels in train:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # parameter specific contribution to changes in the total loss
                    # if the updated parameter is important, the gradient w.r.t. it will be large
                    self.param_updates[name] += param.grad.detach() ** 2

    def consolidate(self):
        for name, param in self.model.named_parameters():
            delta = param.data - self.prev_params[name]
            self.omega[name] += self.param_updates[name] / (delta ** 2 + self.epsilon)
            self.prev_params[name] = param.data.clone()
            self.param_updates[name].zero_()

    def compute_si_loss(self, model: nn.Module, lambda_):
        si_loss = 0
        for name, param in model.named_parameters():
            if name in self.omega:
                si_loss += pt.sum(self.omega[name] * (param - self.prev_params[name]) ** 2)
        return lambda_ * si_loss

# Training loop with SI
def continual_training(model: nn.Module, si, dataset: Dataset, max_epochs: int, loss_list = None, lambda_ = 0.5):
    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 0.1)
    model.train()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(dataset.x)
        loss = criterion(outputs, dataset.y)
        if si is not None:
            loss += si.compute_si_loss(model, lambda_) *  0.005 # 0.005 is a hyperparameter used to scale the SI loss
        loss.backward()
        optimizer.step()
        if loss_list is not None:
            loss_list.append(loss.item())