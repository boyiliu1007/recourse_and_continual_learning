import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.logisticRegression import LogisticRegression
from Models.logisticRegression import training
from Models.synapticIntelligence import SynapticIntelligence, continual_training

train, test, sample = make_dataset(200, 200, 4000)
model = LogisticRegression(train.x.shape[1], 1)
loss_list = []
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
si = SynapticIntelligence(model)

continual_training(model, si, train, 100, loss_list, lambda_=0)

plt.figure()
plt.plot(loss_list)
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss')
plt.savefig('loss_init.png')

si.update_omega(train, criterion)
si.consolidate()