import os
import sys
import matplotlib.pyplot as plt
import cvxpy as cp
import torch as pt
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.strategicClassification import MyStrategicModel, TRAIN_SLOPE, EVAL_SLOPE

train, test, sample = make_dataset(100, 100, 2000)
print(train.x.shape)

XDIM = train.x.shape[1]
COST = 1/XDIM

def score(x, w, b):
    return x@w + b

# here slope is the tamperature parameter
# f is the convex part in the sigmoid
def f(x, w, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, w, b) + 1)]), 2)

# g is the concave part in the sigmoid
def g(x, w, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, w, b) - 1)]), 2)

# c is the cost function (sum of squares/dimensions)
def c(x, r):
    return COST*cp.sum_squares(x-r)

def f_derivative(x, w, b, slope):
    return 0.5*cp.multiply(slope*((slope*score(x, w, b) + 1)/cp.sqrt((slope*score(x, w, b) + 1)**2 + 1)), w)

funcs = {"f": f, "g": g, "f_derivative": f_derivative, "c": c, "score": score}

model = MyStrategicModel(x_dim=XDIM, funcs=funcs, train_slope=TRAIN_SLOPE, eval_slope=EVAL_SLOPE, strategic=True, lamb = 0)

train.x = train.x.type(pt.float64)
train.y = train.y.type(pt.float64)
train.y[train.y == 0] = -1

original = deepcopy(train.y).squeeze()
print(original)

model.fit(train.x, train.y, opt=pt.optim.Adam, opt_kwargs={"lr": 5*(1e-2)}, epochs = 30, batch_size = 100)
with pt.no_grad():
    train.y, tmp, tmp2 = model.forward(train.x, evaluation=True)
train.y = pt.sign(train.y)
print(train.y)

correct_preds = (train.y == original).double()
print(correct_preds)
accuracy = correct_preds.mean().item()

print(f"Accuracy: {accuracy}")

