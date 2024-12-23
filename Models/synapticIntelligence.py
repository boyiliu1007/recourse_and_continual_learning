import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# SI class to manage parameter importance
class SynapticIntelligence:
    def __init__(self, model):
        self.model = model
        self.prev_params = {}
        self.omega = {}
        self.omega_list = []
        self.param_updates = {}
        self.epsilon = 0.00000001  # Small value to avoid division by zero

        for name, param in self.model.named_parameters():
            self.prev_params[name] = param.data.clone()
            self.omega[name] = pt.zeros_like(param)
            self.param_updates[name] = pt.zeros_like(param)

    def update_omega(self, train: Dataset, criterion):
        self.model.eval() # to avoid updating model parameters
        for inputs, labels in train:
            self.model.zero_grad()
            outputs = self.model(inputs)
            if outputs.dim() != labels.dim():
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward() # compute gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # parameter specific contribution to changes in the total loss
                    # if the updated parameter is important, the gradient w.r.t. it will be large
                    self.param_updates[name] += param.grad.detach() ** 2

    # def cosine_similarity_loss(u, v):
    #     dot_prod = pt.sum(u * v, dim=-1)
    #     mag_u = pt.norm(u, dim=-1)
    #     mag_v = pt.norm(v, dim=-1)
    #     cos_sim = dot_prod / (mag_u * mag_v)

    #     # Maximize similarity
    #     loss = 1 - cos_sim

    #     return loss
    
    def _weight_func(self, x):
        return x*x*x
    
    def _get_weight(self, observe_range):
        weights = []
        for i in range (1, observe_range + 1):
            weights.append(self._weight_func(i))
        weights_tensor = pt.tensor(weights, dtype=pt.float32)
        return weights_tensor / sum(weights_tensor)

    def consolidate(self, observe_range):
        curr_omega = {}
        for name, param in self.model.named_parameters():
            delta = param.data - self.prev_params[name]
            curr_omega[name] =  self.param_updates[name] / (delta ** 2 + self.epsilon)
        self.omega_list.append(curr_omega)

        for name, param in self.model.named_parameters():
            if len(self.omega_list) >= observe_range:
                omega_sum = pt.zeros_like(param)
                weights = self._get_weight(observe_range)
                for i in range(-1, -observe_range - 1, -1):
                    omega_sum += self.omega_list[i][name] * weights[i]
                self.omega[name] = omega_sum
            else:
                for i in range(len(self.omega_list)):
                    self.omega[name] += self.omega_list[i][name]
            self.prev_params[name] = param.data.clone()
            self.param_updates[name].zero_()

    # def consolidate(self):
    #     for name, param in self.model.named_parameters():
    #         delta = param.data - self.prev_params[name]
    #         self.omega[name] += self.param_updates[name] / (delta ** 2 + self.epsilon)
    #         self.prev_params[name] = param.data.clone()
    #         self.param_updates[name].zero_()

    def compute_si_loss(self, lambda_):
        si_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                # print("param", param)
                # print("prev_params", self.prev_params[name])
                si_loss += pt.sum(self.omega[name] * (param - self.prev_params[name]) ** 2)
        return lambda_ * si_loss

# Training loop with SI
def continual_training(si: SynapticIntelligence, dataset: Dataset, max_epochs: int, loss_list = None, lambda_ = 0.5):
    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(si.model.parameters(), 0.1,weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    si.model.train()
    epoch_loss = []
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for _ in range(max_epochs):
        running_loss = 0.0
        for X_batch,Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = si.model(X_batch)
            outputs = outputs.squeeze()
            if outputs.dim() != Y_batch.dim():
                    outputs = outputs.unsqueeze(-1)
            loss = criterion(outputs, Y_batch)
            loss += si.compute_si_loss(lambda_) * 0.005 # 0.005 is a hyperparameter used to scale the SI loss
            running_loss += loss.item() * X_batch.size(0)
            loss.backward()
            optimizer.step()
            if loss_list is not None:
                loss_list.append(loss.item())
        
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        epoch_loss.append(train_loss)

        if ( ( (_+1) % 15 == 0) | (_ == 0) ) :
            print(f"Epoch {_+1}/{max_epochs}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")






