import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.logisticRegression import LogisticRegression
from Models.logisticRegression import training

<<<<<<< HEAD
POSITIVE_RATIO = 0.5  #5000,1500,7500
train, test, sample, dataset = make_dataset(2000, 500, 7500,POSITIVE_RATIO,'UCIcredit')#2000,500,7500
=======
POSITIVE_RATIO = 0.4  #5000,1500,7500
train, test, sample, dataset = make_dataset(700, 500, 2500,POSITIVE_RATIO,'synthetic')#2000,500,7500
>>>>>>> b5e9c10976e71faeacf1c33a0d21e25dae1abc37
# train, test, sample, dataset = make_dataset(1000, 250, 5000,POSITIVE_RATIO,'credit')
# train, test, sample = make_dataset(100, 100, 2000,POSITIVE_RATIO)
print(f"train.x.shape: {train.x.shape}")
model = LogisticRegression(train.x.shape[1], 1)
loss_list = []
training(model, train, 50, test,loss_list)

plt.figure()
plt.plot(loss_list)
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss')
plt.savefig('logisticRegression_loss_init.png')

