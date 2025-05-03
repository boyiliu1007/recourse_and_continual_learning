import sys
import matplotlib.pyplot as plt
import torch as pt
import math
import os
import pandas as pd
import datetime
from sklearn.datasets import make_classification
os.makedirs("Pure Recourse Experiment/Results", exist_ok=True)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.LogisticRegression import LogisticRegression, training
from Dataset.makeDataset import Dataset
from Model.Recourse import recourse
from Test.test_recourse_only_modify_selected import test_only_selected_data_modified
from Auxiliary.plot_decision_boundary import plot_decision_boundary_pca, plot_decision_boundary_tsne, plot_decision_boundary_umap, plot_decision_boundary


# generate dataset
DIMENSIONS = 2
POSITIVE_RATIO = 0.5 
# x, y = make_classification(
#     n_samples=128, 
#     n_features=DIMENSIONS,
#     n_informative=DIMENSIONS,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     weights=[1 - POSITIVE_RATIO, POSITIVE_RATIO],
#     random_state=47
# )
# x = pt.tensor(x, dtype=pt.float)
# y = pt.tensor(y[..., None], dtype=pt.float).squeeze()

# num_points = 64
# x = pt.randn(num_points, 2)
# x_sym = x.flip(1)
# x_full = pt.cat([x, x_sym], dim=0)
# y_full = pt.cat([pt.zeros(num_points), pt.ones(num_points)])

sqrt3 = pt.sqrt(pt.tensor(3.0))
R = pt.tensor([
    [-0.5, sqrt3 / 2],
    [sqrt3 / 2, 0.5]
])

# Step 1: Random 2D points
num_points = 100
x = pt.randn(num_points, 2)

# Step 2: Reflect across y = 2x
x_reflected = x_reflected = x @ R.T

# Step 3: Combine original + reflected
x_full = pt.cat([x, x_reflected])
y_full = pt.cat([pt.zeros(num_points), pt.ones(num_points)])

train = Dataset(x_full, y_full)
model = LogisticRegression(train.x.shape[1], 1)

# initial model
loss_list = []
# model = training(model, train, 200, loss_list)
# random state = 43
model.linear.weight.data[0][0] = -1.732
model.linear.weight.data[0][1] = 1
model.linear.bias.data[0] = 0
y_pred = model(train.x).detach().squeeze()
train.y = pt.where(y_pred > 0.5, 1, 0).float()

# random state = 47
# model.linear.weight.data[0][0] = 4.9014
# model.linear.weight.data[0][1] = 4.9014
# model.linear.bias.data[0] = 4.9516
# y_pred = model(train.x).detach().squeeze()
# train.y = pt.where(train.y > 0.5, 1, 0).float()
low_cost_model_shift = 0
high_cost_model_shift = 0
prev_model = None
bias_list = []
cosine_list = []
# plot_decision_boundary_pca(model, train, "Pure Recourse Experiment/Results/recourse_round_0.png", DIMENSIONS)
# plot_decision_boundary_umap(train, "Pure Recourse Experiment/Results/recourse_round_0_umap.png", DIMENSIONS)
# plot_decision_boundary_tsne(model, train, "Pure Recourse Experiment/Results/recourse_round_0_tsne.png")
plot_decision_boundary(model, train, "Pure Recourse Experiment/Results/recourse_round_0.png", 0)
print("model weight:", model.linear.weight.data) # debug log
print("model bias:", model.linear.bias.data) # debug log
for i in range(10):
    print(f"round: {i+1}")
    # norm 
    # train.x = (train.x - train.x.mean(0)) / train.x.std(0)
    # create this for testing to ensure recourse function is correct
    before_recoursed = Dataset(
        train.x.clone().detach(), 
        train.y.clone().detach()
    )

    # find training data with label 0 and select 0.5 of them
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
    # print("act", act)
    train.x[selected_indices] = recoursed_data.x.detach()
    train.y[selected_indices] = recoursed_data.y.detach()
    
    # test recourse function
    if test_only_selected_data_modified(model, before_recoursed, train, selected_indices) != True:
        print("test error")


    # topk method
    with pt.no_grad():
        y_prob_all: pt.Tensor = model(train.x)
    sorted_indices = pt.argsort(y_prob_all, dim=0, descending=True)
    cutoff_index = int(len(sorted_indices) * POSITIVE_RATIO)
    mask = pt.zeros_like(train.y)
    mask[sorted_indices[:cutoff_index].squeeze()] = 1
    train.y = mask.float()

    prev_model = model.linear.weight.data.clone()
    
    # update model
    model = training(model, train, 200, loss_list)

    weight = model.linear.weight.data[0]
    bias = model.linear.bias.data.item()
    bias_list.append(bias)
    slope = -weight[0].item() / weight[1].item()
    print(f"Decision boundary slope: {slope:.4f}")
    # calculate cosine similarity between the two weight vectors
    cosine_similarity = pt.nn.functional.cosine_similarity(prev_model, model.linear.weight.data[0])
    cosine_list.append(cosine_similarity.item())
    print(f"Cosine similarity: {cosine_similarity.item():.4f}")
    print("model weight:", model.linear.weight.data) # debug log
    # print(model.linear.bias.data)
    print("model shift on low cost", pt.abs(model.linear.weight.data[0][0] - prev_model[0][0]).item())
    print("model shift on high cost", pt.abs(model.linear.weight.data[0][1] - prev_model[0][1]).item())
    low_cost_model_shift += pt.abs(model.linear.weight.data[0][0] - prev_model[0][0]).item()
    high_cost_model_shift += pt.abs(model.linear.weight.data[0][1] - prev_model[0][1]).item()
    plot_path = f"Pure Recourse Experiment/Results/recourse_round_{i+1}.png"
    # plot_decision_boundary_umap(train, plot_path, DIMENSIONS)
    # plot_decision_boundary_pca(model, train, plot_path, DIMENSIONS)
    # plot_decision_boundary_tsne(model, train, plot_path)
    plot_decision_boundary(model, train, plot_path, i)
    print("====================================")

print("overall low cost model shift", low_cost_model_shift)
print("overall high cost model shift", high_cost_model_shift)
print("bias list", bias_list)
print("cosine list", cosine_list)

# store the bias list and cosine list to a csv file
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

bias_df = pd.DataFrame(bias_list, columns=["bias"])
cosine_df = pd.DataFrame(cosine_list, columns=["cosine"])

# bias_df.to_csv(f"Pure Recourse Experiment/Results/bias_list_{current_time}.csv", index=False)
# cosine_df.to_csv(f"Pure Recourse Experiment/Results/cosine_list_{current_time}.csv", index=False)
