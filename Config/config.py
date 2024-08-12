import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.logisticRegression import LogisticRegression
from Models.logisticRegression import training

POSITIVE_RATIO = 0.75
train, test, sample = make_dataset(100, 100, 2000, POSITIVE_RATIO)
print(train.x.shape)
model = LogisticRegression(train.x.shape[1], 1)
loss_list = []
training(model, train, 50, loss_list)

plt.figure()
plt.plot(loss_list)
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss')
plt.savefig('logisticRegression_loss_init.png')

