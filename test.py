import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class A:
    def __init__(self):
        self.model = SimpleModel()

def modify_weights(model):
    # Access and modify the weights of the first layer (fc)
    with torch.no_grad():  # Disable gradient tracking to modify weights
        model.fc.weight.fill_(2.0)  # Set all weights to 2.0
        model.fc.bias.fill_(1.0)    # Set all biases to 1.0

test = A()
print("Original weights: ", test.model.fc.weight)
print("Original bias: ", test.model.fc.bias)
modify_weights(test.model)
print("Modified weights: ", test.model.fc.weight)
print("Modified bias: ", test.model.fc.bias)
