import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.logisticRegression import LogisticRegression
from Models.logisticRegression import training

POSITIVE_RATIO = 0.5  #5000,1500,7500
train, test, sample, dataset = make_dataset(700, 500, 2500,POSITIVE_RATIO,'synthetic')#2000,500,7500


# train, test, sample, dataset = make_dataset(1000, 250, 5000,POSITIVE_RATIO,'credit')
# train, test, sample = make_dataset(100, 100, 2000,POSITIVE_RATIO)
print(f"train.x.shape: {train.x.shape}")
model = LogisticRegression(train.x.shape[1], 1)
loss_list = []
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train.x, train.y.ravel())  # Make sure y is 1D

# Get feature importances
importances = rf.feature_importances_

# Rank features
ranked_features = np.argsort(-importances)

# Print in order
for idx in ranked_features:
    print(f"Feature {idx}: Importance {importances[idx]:.4f}")
training(model, train, 50, test,loss_list)

plt.figure()
plt.plot(loss_list)
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss')
plt.savefig('logisticRegression_loss_init.png')

