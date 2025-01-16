import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.MLP import MLP
from Models.MLP import training

POSITIVE_RATIO = 0.5
# train, test, sample = make_dataset(2000, 500, 7500,POSITIVE_RATIO)
# train, test, sample = make_dataset(1000, 250, 5000,POSITIVE_RATIO)
train, test, sample, dataset = make_dataset(1000, 250, 5000,POSITIVE_RATIO)
# train, test, sample = make_dataset(100, 100, 2000,POSITIVE_RATIO)
print(f"train.x.shape: {train.x.shape}")
model = MLP(train.x.shape[1], 1)
loss_list = []
training(model, train, 50, test,loss_list)

plt.figure()
plt.plot(loss_list)
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss')
plt.savefig('MLP_loss_init.png')

