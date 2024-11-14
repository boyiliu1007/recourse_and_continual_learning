from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import pandas as pd
import os
import time
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import cvxpy as cp
import torch as pt
from cvxpylayers.torch import CvxpyLayer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset

TRAIN_SLOPE = 1
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class CCP:
    def __init__(self, x_dim, funcs):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)
        print(self.prob.is_dcp(dpp=True))
        
    def ccp(self, r):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.0001 and cnt < 10:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value
    
    def optimize_X(self, X, w, b, slope):
        """
        tensor to tensor
        """
        w = w.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()
        
        self.w.value = w
        self.b.value = b
        self.slope.value = slope
        
        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])

class DELTA():
    def __init__(self, x_dim, funcs):
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        # problem
        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]

class MyStrategicModel(torch.nn.Module):
    def __init__(self, x_dim, funcs, train_slope, eval_slope, strategic=False, lamb=0):
        torch.manual_seed(0)
        np.random.seed(0)
        super(MyStrategicModel, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        self.sigmoid = torch.nn.Sigmoid()
        self.strategic = strategic
        self.lamb = lamb
        self.ccp = CCP(x_dim, funcs)
        self.delta = DELTA(x_dim, funcs)

    def forward(self, X, evaluation=False):
        # problem
        if evaluation:
            XT = self.ccp.optimize_X(X, self.w, self.b, self.eval_slope)
            X_opt = XT
        else:
            XT = self.ccp.optimize_X(X, self.w, self.b, self.train_slope)
            F_DER = self.get_f_ders(XT, self.train_slope)
            X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER) # Xopt should equal to XT but we do it again for the gradients
        
        recourse = self.calc_recourse(X, X_opt)        
        if self.strategic:
            output = self.score(X_opt)
        else:
            output = self.score(X)        
        
        return output, recourse, X_opt
    
    def optimize_X(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        return self.ccp.optimize_X(X, self.w, self.b, slope)
    
    def score(self, x):
        return x@self.w + self.b
    
    def get_f_ders(self, XT, slope):
        return torch.stack([0.5*slope*((slope*self.score(xt) + 1)/torch.sqrt((slope*self.score(xt) + 1)**2 + 1))*self.w for xt in XT])

    def evaluate(self, X, Y):
        scores, _ = self.forward(X, evaluation=True)
        Y_pred = torch.sign(scores)
        num = len(Y)
        temp = Y - Y_pred
        acc = len(temp[temp == 0])*1./num        
        return acc
        
    def calc_recourse(self, X, X_opt):      
        S = self.score(X)
        is_neg = self.sigmoid(-S)
        
        S = self.score(X_opt)
        is_not_able_to_be_pos = self.sigmoid(-S)
        
        recourse_penalty = is_neg * is_not_able_to_be_pos
        return 1 - torch.mean(recourse_penalty)
    
    def loss(self, Y, Y_pred, recourse):
        if self.lamb > 0:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0)) + self.lamb*(1 - recourse)
        else:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))
        
    
    # def save_model(self, X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_recourses, info, path, comment=None):
    #     if comment is not None:
    #         path += "_____" + comment
            
    #     filename = path + "/model.pt"
    #     if not os.path.exists(os.path.dirname(filename)):
    #         os.makedirs(os.path.dirname(filename))
    #     torch.save(self.state_dict(), filename)
        
    #     pd.DataFrame(X.numpy()).to_csv(path + '/X.csv')
    #     pd.DataFrame(Y.numpy()).to_csv(path + '/Y.csv')
    #     pd.DataFrame(Xval.numpy()).to_csv(path + '/Xval.csv')
    #     pd.DataFrame(Yval.numpy()).to_csv(path + '/Yval.csv')
    #     pd.DataFrame(Xval.numpy()).to_csv(path + '/Xtest.csv')
    #     pd.DataFrame(Yval.numpy()).to_csv(path + '/Ytest.csv')
        
    #     pd.DataFrame(np.array(train_errors)).to_csv(path + '/train_errors.csv')
    #     pd.DataFrame(np.array(val_errors)).to_csv(path + '/val_errors.csv')
    #     pd.DataFrame(np.array(train_losses)).to_csv(path + '/train_losses.csv')
    #     pd.DataFrame(np.array(val_losses)).to_csv(path + '/val_losses.csv')
    #     pd.DataFrame(np.array(val_recourses)).to_csv(path + '/val_recourses.csv')
        
    #     with open(path + "/info.txt", "w") as f:
    #         f.write(info)
    
    # def load_model(self, filename):
    #     self.load_state_dict(torch.load(filename))
    #     self.eval()
    
    def fit(self, X, Y, opt, opt_kwargs={"lr":1e-3}, batch_size=1, epochs=30, verbose=True):
        train_dset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        opt = opt(self.parameters(), **opt_kwargs)

        epoch_train_losses = []
        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []
        # val_recourses = []
        
        # best_val_error = 1
        # best_val_recourse = 1
        # consecutive_no_improvement = 0
        # now = datetime.now()
        # path = "./models/recourse/" + now.strftime("%d-%m-%Y_%H-%M-%S")

        # total_time = time.time()
        for epoch in range(epochs):
            print("epoch: ", epoch)
            # t1 = time.time()
            batch = 1
            train_losses.append([])
            tensor_list = []
            # train_errors.append([])
            for Xbatch, Ybatch in train_loader:
                opt.zero_grad()
                Ybatch_pred, recourse , tmpX = self.forward(Xbatch)
                tensor_list.append(tmpX)
                l = self.loss(Ybatch, Ybatch_pred, recourse)
                l.backward()
                opt.step()
                train_losses[-1].append(l.item())
                # if calc_train_errors:
                #     with torch.no_grad():
                #         e = self.evaluate(Xbatch, Ybatch)
                #         train_errors[-1].append(1-e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | recourse: %3.5f" %
                          (batch, len(train_loader), np.mean(train_losses[-1]), recourse))
                batch += 1
                # if callback is not None:
                #     callback()

            # ======================= I add  this ====================
            result_tensor = torch.cat(tensor_list, dim=0)
            # print(result_pred.shape, result_tensor.shape)
            epoch_train_losses.append(np.mean(train_losses[-1]))

        plt.figure()
        plt.plot(range(0, epochs), epoch_train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()
        # ============================================================

            # with torch.no_grad():
            #     Yval_pred, val_recourse = self.forward(Xval, evaluation=True)
            #     val_recourse = val_recourse.item()
            #     val_loss = self.loss(Yval, Yval_pred, val_recourse).item()
            #     val_losses.append(val_loss)
            #     val_error = 1-self.evaluate(Xval, Yval)
            #     val_errors.append(val_error)
            #     val_recourses.append(val_recourse)
            #     if val_error < best_val_error:
            #         consecutive_no_improvement = 0
            #         best_val_error = val_error
            #         # if self.strategic:
            #         #     info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\nrecourse: {}".format(
            #         #     time.time()-total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], val_loss, val_error, val_recourse)
            #         #     # self.save_model(X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_recourses, info, path, comment)
            #         #     print("model saved!")
            #     else:
            #         consecutive_no_improvement += 1
            #         if consecutive_no_improvement >= 6:
            #             break
                
            # t2 = time.time()
            # if verbose:
                # print("----- epoch %03d / %03d | time: %03d sec | loss: %3.5f | recourse: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_recourses[-1], val_errors[-1]))
        # print("training time: {} seconds".format(time.time()-total_time)) 
        return result_tensor
    

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