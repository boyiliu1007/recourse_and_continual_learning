import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# SI class implementing the original algorithm from
# "Continual Learning Through Synaptic Intelligence"
class SynapticIntelligence:
    def __init__(self, model):
        self.model = model
        self.prev_params = {}  # Previous parameter values
        self.initial_params = {}  # Parameters at the start of a task
        self.omega = {}  # Importance parameters
        self.omega_list = []  # Importance parameters for all previous tasks
        self.path_integrals = {}  # Accumulates -grad * parameter_change
        self.epsilon = 1e-8  # Small value to avoid division by zero

        # Initialize tracking variables for all parameters
        for name, param in self.model.named_parameters():
            self.prev_params[name] = param.data.clone()
            self.initial_params[name] = param.data.clone()
            self.omega[name] = pt.zeros_like(param)
            self.path_integrals[name] = pt.zeros_like(param)

    def update_path_integral(self):
        """
        Update the path integral during training.
        Call this BEFORE each optimizer step.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Current parameter values
                curr_param = param.data.clone()
                # Parameter change from last update
                delta_param = curr_param - self.prev_params[name]
                # Update path integral with negative gradient * parameter change
                self.path_integrals[name] -= param.grad.detach() * delta_param
                # Store current parameters for next update
                self.prev_params[name] = curr_param

    def consolidate(self):
        """
        Compute omega values and finalize importance after a task is complete.
        Call this at the end of each task's training.
        """
        current_task_omega = {}
        
        # Calculate importance for each parameter
        for name, param in self.model.named_parameters():
            # Total parameter change from task start to end
            total_change = param.data - self.initial_params[name]
            # Normalize path integral by square of total change
            denominator = total_change**2 + self.epsilon
            current_task_omega[name] = self.path_integrals[name] / denominator
            
            # Add to cumulative importance (original paper suggests sum)
            if name in self.omega:
                self.omega[name] += current_task_omega[name]
            else:
                self.omega[name] = current_task_omega[name]
        
        # Keep track of per-task omegas for reference
        self.omega_list.append(current_task_omega)
        
        # Reset for next task
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.data.clone()
            self.prev_params[name] = param.data.clone()
            self.path_integrals[name].zero_()

    def compute_si_loss(self, lambda_):
        """
        Compute the SI regularization loss to prevent forgetting.
        lambda_ controls the strength of regularization.
        """
        si_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                # Penalize changes to important parameters
                si_loss += pt.sum(self.omega[name] * (param - self.prev_params[name])**2)
        return lambda_ * si_loss


# Training loop with SI
def continual_training(si: SynapticIntelligence, dataset: Dataset, max_epochs: int, loss_list=None, lambda_=0.5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(si.model.parameters(), 0.1, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    si.model.train()
    epoch_loss = []
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    for _ in range(max_epochs):
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = si.model(X_batch)
            outputs = outputs.squeeze()
            
            if outputs.dim() != Y_batch.dim():
                outputs = outputs.unsqueeze(-1)
            
            # Calculate loss with SI regularization
            task_loss = criterion(outputs, Y_batch)
            si_loss = si.compute_si_loss(lambda_)
            loss = task_loss + si_loss
            running_loss += loss.item() * X_batch.size(0)
            
            # Backward pass
            loss.backward()
            
            # Update path integral with current gradients
            si.update_path_integral()
            
            # Update parameters
            optimizer.step()
            
            if loss_list is not None:
                loss_list.append(loss.item())
        
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        epoch_loss.append(train_loss)

        if ((_+1) % 15 == 0) or (_ == 0):
            print(f"Epoch {_+1}/{max_epochs}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # After task training is complete, consolidate knowledge
    si.consolidate()
    