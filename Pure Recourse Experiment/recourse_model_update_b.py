import sys
import matplotlib.pyplot as plt
import torch as pt
import math
import os
from sklearn.datasets import make_classification
os.makedirs("Pure Recourse Experiment/Results", exist_ok=True)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.LogisticRegression import LogisticRegression, training_update_bias_only, training
from Dataset.makeDataset import Dataset
from Model.Recourse import recourse
from Test.test_recourse_only_modify_selected import test_only_selected_data_modified
from Auxiliary.plot_decision_boundary import plot_decision_boundary

# generate dataset
POSITIVE_RATIO = 0.5 
x, y = make_classification(
    n_samples=128, 
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[1 - POSITIVE_RATIO, POSITIVE_RATIO],
    random_state=42
)
x = pt.tensor(x, dtype=pt.float)
y = pt.tensor(y[..., None], dtype=pt.float).squeeze()

train = Dataset(x, y)
model = LogisticRegression(train.x.shape[1], 1)

# initial model
loss_list = []
model = training(model, train, 200, loss_list)
plot_decision_boundary(model, train, "Pure Recourse Experiment/Results/recourse_round_0.png")

for i in range(10):
    print(f"round: {i}")
    # create this for testing to ensure recourse function is correct
    before_recoursed = Dataset(
        train.x.clone().detach(), 
        train.y.clone().detach()
    )

    # find training data with label 0 and select 0.1 of them
    data, labels = train.x, train.y
    label_0_indices = pt.where(labels == 0)[0]
    shuffled_indices = pt.randperm(len(label_0_indices))
    label_0_indices = label_0_indices[shuffled_indices]
    num_samples = math.floor(len(label_0_indices) * 0.5)
    selected_indices = label_0_indices[:num_samples]
    print(f"{num_samples} in train data do recourse")

    selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))

    # update train data with recoursed data
    recoursed_data, act = recourse(model, selected_subset, 200)
    print(act)
    train.x[selected_indices] = recoursed_data.x.detach()
    train.y[selected_indices] = recoursed_data.y.detach()

    # test recourse function
    test_only_selected_data_modified(before_recoursed, train, selected_indices)

    model = training_update_bias_only(model, train, 200, loss_list)
    plot_path = f"Pure Recourse Experiment/Results/recourse_round_{i+1}.png"
    plot_decision_boundary(model, train, plot_path)
    print("====================================")


