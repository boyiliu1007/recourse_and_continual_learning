from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.colors import ListedColormap
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.patches import Rectangle
import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Config.continual_config import test, train, sample
from Models.synapticIntelligence import SynapticIntelligence

pca = PCA(2).fit(train.x)

class Helper:

    palette = sns.color_palette('muted', 2)
    cmap = ListedColormap(palette)

    def __init__(self, model: nn.Module, pca: PCA, train: Dataset, test: Dataset, sample: Dataset):
        self.model = model
        self.pca = pca
        self.train = train
        self.test = test
        self.sample = sample
        self.PDt = []
        self.round = 0
        self.EFTdataframe =  pd.DataFrame(
            {
                'x':train.x.tolist(),
                'y':train.y.flatten(),
                # 'Predict':train.y.flatten(),
                'Predict':[[] for _ in range(len(train.x))],
                'flip_times':np.zeros(len(train.x)),
                'startRounds':np.zeros(len(train.x)),
                'updateRounds':np.zeros(len(train.x)),
                'EFT' : np.zeros(len(train.x)),
                'EFTList': [[] for _ in range(len(train.x))]
            }
        )
        self.failToRecourseOnModel = []
        self.failToRecourseOnLabel = []
        self.failToRecourse = []

        self.validation_list = []
        self.Ajj_performance_list = []
        self.overall_acc_list = []
        self.memory_stability_list = []
        self.memory_plasticity_list = []
        self.Aj_tide_list = []
        self.jsd_list = []

        self._hist: list[BarContainer]
        self._bins: NDArray
        self._sc_train: PathCollection
        self._sc_test: PathCollection
        self._ct_test: QuadContourSet
        self.lr = 0.1
        self.si = SynapticIntelligence(self.model)
        self.save_directory = None

    # def draw_proba_hist(self, ax: Axes | None = None, *, label: bool = False):
    def draw_proba_hist(self, ax: Axes = None, *, label: bool = False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        x = self.test.x
        y = self.test.y

        n = x.shape[0]
        m = n - pt.count_nonzero(y)

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x)

        y = y.flatten()
        y_prob = y_prob.flatten()
        y_prob = y_prob[y.argsort()]

        w = np.broadcast_to(100 / n, n)

        self._hist: list[BarContainer]
        _, self._bins, self._hist = ax.hist(
            (y_prob[:m], y_prob[m:]),
            10,
            (0, 1),
            weights=(w[:m], w[m:]),
            rwidth=1,
            color=self.palette,
            label=(0, 1),
            ec='w',
            alpha=0.9,
        )

        ax.legend(loc='upper center', title='True label')
        ax.set_xlabel('mean predicted probabilty')
        ax.set_ylabel('percent')
        ax.set_title('Probability Distribution')

        if label:
            for c in self._hist:
                height = map(Rectangle.get_height, c.patches)
                ax.bar_label(
                    c,
                    [f'{h}%' if h else '' for h in height],
                    fontsize='xx-small'
                )
        return fig, ax

    #calculate js divergence using training data after pca
    def js_divergence(self, pcaData, labelData):
        data1 = pcaData[labelData.flatten() == 0]
        data2 = pcaData[labelData.flatten() == 1]

        # Fit KDEs to the reduced data
        kde1 = KernelDensity(kernel='gaussian', bandwidth=1).fit(data1)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=1).fit(data2)

        # Evaluate KDEs on a grid of points
        grid_points = np.random.randn(1000, 2)  # 1000 points in 2D space
        log_dens1 = kde1.score_samples(grid_points)
        log_dens2 = kde2.score_samples(grid_points)

        # Convert to densities
        dens1 = np.exp(log_dens1)
        dens2 = np.exp(log_dens2)

        js_divergence = jensenshannon(dens1, dens2)
        print("js_divergence: ",js_divergence)
        self.jsd_list.append(js_divergence)

    # def draw_dataset_scatter(self, axes: tuple[Axes, Axes] | None = None):
    def draw_dataset_scatter(self, axes: [Axes, Axes] = None):
        if axes is None:
            fig, (ax0, ax1) = plt.subplots(
                1, 2,
                sharex=True,
                sharey=True,
                figsize=(8, 4),
                layout='compressed'
            )
        else:
            ax0, ax1 = axes
            fig = ax0.get_figure()

        prop = dict(cmap=self.cmap, s=40, vmin=0., vmax=1., lw=0.8, ec='w')

        self._sc_train = ax0.scatter(
            *pca.transform(self.train.x).T,
            c=self.train.y,
            **prop
        )
        ax0.legend(
            *self._sc_train.legend_elements(),
            loc='upper right',
            title='True label'
        )
        ax0.set_xlabel('pca0')
        ax0.set_ylabel('pca1')
        ax0.set_title('Train')

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(test.x)

        y_prob = y_prob.flatten()
        y_pred = y_prob.greater(0.5)

        self._sc_test = ax1.scatter(
            *pca.transform(test.x).T,
            c=y_pred,
            **prop
        )
        ax1.legend(
            *self._sc_test.legend_elements(),
            loc='upper right',
            title='Predicted'
        )

        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        n = 32
        xy = np.mgrid[x0: x1: n * 1j, y0: y1: n * 1j]
        z = pca.inverse_transform(xy.reshape(2, n * n).T)

        z = pt.tensor(z, dtype=pt.float)
        with pt.no_grad():
            z: pt.Tensor = self.model(z)
        z = z.view(n, n)
        self._ct_test = ax1.contourf(
            *xy, z, 10,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            alpha=0.9,
            zorder=0,
        )
        fig.colorbar(self._ct_test, ax=ax1, label='probability')
        ax1.grid(alpha=0.75)
        ax1.set_xlabel('pca0')
        ax1.set_title('Test')

        return fig, axes

    def draw_all(self):
        sf: list[SubFigure]
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        sf = fig.subfigures(2, 1)
        ax0 = sf[0].subplots()
        ax1, ax2 = sf[1].subplots(1, 2, sharex=True, sharey=True)
        self.draw_proba_hist(ax0)
        self.draw_dataset_scatter((ax1, ax2))
        return fig, (ax0, ax1, ax2)

    def animate_all(self, frames: int = 120, fps: int = 10, *, inplace: bool = False):
        fig, (ax0, ax1, ax2) = self.draw_all()

        model = self.model if inplace else deepcopy(self.model)
        train = self.train if inplace else deepcopy(self.train)
        test = self.test
        sample = self.sample

        def init():
            return *ax0.patches, *ax1.collections, *ax2.collections

        def func(frame):
            fig.suptitle(f'No. {frame}', ha='left', x=0.01, size='small')

            if frame == 0:
                return ()

            self.update(model, train, sample)

            y = test.y.flatten()

            n = test.x.shape[0]
            m = n - pt.count_nonzero(y)

            with pt.no_grad():
                y_prob: pt.Tensor = model(test.x)

            y_prob = y_prob.flatten()
            y_pred = y_prob.greater(0.5)
            rank = y_prob[y.argsort()]

            for b, r in zip(self._hist, (rank[:m], rank[m:])):
                height, _ = np.histogram(r, self._bins)
                for r, h in zip(b.patches, height * (100 / n)):
                    r.set_height(h)

            self._sc_train.set_offsets(pca.transform(train.x))
            self._sc_train.set_array(train.y.flatten())

            # calculate js divergence of pca training data
            self.js_divergence(self._sc_train.get_offsets(), self._sc_train.get_array())
            self._sc_test.set_array(y_pred)

            ax1.relim()
            ax1.autoscale_view()

            for c in self._ct_test.collections:
                c.remove()

            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            n = 32
            xy = np.mgrid[x0: x1: n * 1j, y0: y1: n * 1j]
            z = pca.inverse_transform(xy.reshape(2, n * n).T)
            z = pt.tensor(z, dtype=pt.float)
            with pt.no_grad():
                z: pt.Tensor = model(z)
            z = z.view(n, n)

            self._ct_test: QuadContourSet = ax2.contourf(
                *xy, z, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )

            return *ax0.patches, *ax1.collections, *ax2.collections

        return FuncAnimation(
            fig, func, frames, init,
            interval=1000 // fps,
            repeat=False,
            blit=True,
            cache_frame_data=False
        )

    def draw_PDt(self):

        #每一個round算出來的L2 distance
        tempList = deepcopy(self.PDt)
        # print("PDt: ",self.PDt)
        # print("tempList: ",tempList)
        # [1,2,3,4,5]
        for index,i in enumerate(range(len(self.PDt))):
            temp = 0.0
            #算出PDt
            for j in range(i + 1):
                temp = temp + tempList[j]
            temp = temp / (index + 1)
            # print("temp: ",temp)
            # print(type(temp))
            self.PDt[index] = temp

        # print("self.PDt: ",self.PDt)

        plt.figure()
        plt.plot(self.PDt)
        plt.xlabel('Round')
        plt.ylabel('PDt')
        plt.title('PDt during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'PDt during Rounds.png'))

    def draw_EFT(self,epochs):
        data = []
        labels = [(epochs / 10 - 1) * (i+1) for i in range(10)]
        x = [1,2,3,4,5,6,7,8,9,10]
        print("EFT labels: ",labels)
        # labels = [8 * (i+1) for i in range(12)]
        # x = [1,2,3,4,5,6,7,8,9,10,11,12]
        for i in range((int(epochs / 10)-1),int(labels[-1]) + 1,(int(epochs / 10) - 1)):
        # for i in range(8,100,8):
            # print("i",i)
            roundData = []
            #只顯示經過2輪以上update的model的資料
            # dataframe = self.EFTdataframe[self.EFTdataframe['updateRounds'] > 2]
            dataframe = self.EFTdataframe[i - self.EFTdataframe['startRounds'] >= 2]
            # dataframe = self.EFTdataframe[self.EFTdataframe['flip_times'] > 0].head(10)
            for j in dataframe.index:
                # print("index: ",j)
                #判斷開始的Round是否合理
                if dataframe.at[j,'startRounds'] < i:
                    # print("index j append: ",j)
                    temp = dataframe.at[j,'EFTList']
                    roundData.append(temp[i - 1])
                # data.append([temp[i]])
                # self.EFTdataframe.at[i,'EFPList'].append(self.EFTdataframe.at[i,'flip_times'] / self.EFTdataframe.at[i,'rounds'])
            # print(data)
            data.append(roundData)
        plt.figure()
        # sns.boxplot(x='rounds',y='EFPList',data = self.EFTdataframe)
        plt.boxplot(data)
        plt.xticks(x,labels)
        plt.xlabel('Round')
        plt.ylabel('EFT')
        plt.savefig(os.path.join(self.save_directory, 'EFT.png'))

    def calculateR20EFT(self,tempPredict,updateRounds):
        last = tempPredict[0]
        flipTimes = 0
        for i in range(len(tempPredict) - 1):
            if(last != tempPredict[i + 1]):
                flipTimes += 1
            last = tempPredict[i + 1]
        # print("R20_EFT value: ",flipTimes / updateRounds)
        return flipTimes / updateRounds

    def draw_R20_EFT(self,epochs,intervalRounds):
        data = []
        labels = [i for i in range(intervalRounds,epochs,intervalRounds)]
        x = [i for i in range(1,len(labels) + 1)]
        # labels = [(int(epochs / 10) - 1) * (i+1) for i in range(10)]
        # x = [1,2,3,4,5,6,7,8,9,10]

        # for i in range((int(epochs / 10)-1),epochs - 1,(int(epochs / 10) - 1)):
        for i in range(intervalRounds,epochs - 1,intervalRounds):
            # print("i's round:",i)
            roundData = []
            # dataframe = self.EFTdataframe[self.EFTdataframe['updateRounds'] > 2]
            dataframe = self.EFTdataframe[i - self.EFTdataframe['startRounds'] >= 2]
            # print("R20's EFTdataframe")
            # display(dataframe)
            for j in dataframe.index:
                startRounds = int(dataframe.at[j,'startRounds'])
                if startRounds < i:
                    #dataframe['Predict'][0]代表從第predictRound開始做預測
                    # print("startRound : ",startRounds)

                    if startRounds >= i - intervalRounds:
                        startIndex = 0
                    else:
                        # startIndex = ((i - (int(epochs / 10) - 1)) - startRounds) - 1
                        startIndex = ((i - intervalRounds) - startRounds) - 1

                    endIndex = (i - startRounds)
                    # print("Predict array: ",dataframe.at[j,'Predict'])
                    # print("startIndex: ",startIndex)
                    # print("endIndex : ",endIndex)
                    tempPredict = dataframe.at[j,'Predict'][startIndex:endIndex]

                    roundData.append(self.calculateR20EFT(tempPredict,len(tempPredict)))


            data.append(roundData)
        plt.figure()
        # sns.boxplot(x='rounds',y='EFPList',data = self.EFTdataframe)
        plt.boxplot(data)
        plt.xticks(x,labels)
        plt.xlabel('Round')
        plt.ylabel('R' + str(intervalRounds) + '_EFT')
        plt.savefig(os.path.join(self.save_directory, f'R{intervalRounds}_EFT.png'))

    def draw_Fail_to_Recourse(self):
        plt.figure()
        plt.plot(self.failToRecourse)
        plt.xlabel('Round')
        plt.ylabel('Fail_to_Recourse')
        plt.title('FtR during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'Fail_to_Recourse.png'))

    def draw_Fail_to_Recourse_on_Model(self):
        plt.figure()
        plt.plot(self.failToRecourseOnModel)
        plt.xlabel('Round')
        plt.ylabel('Fail_to_Recourse_on_Model')
        plt.title('FtR during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'Fail_to_Recourse_on_Model.png'))

    def draw_Fail_to_Recourse_on_Label(self):
        plt.figure()
        plt.plot(self.failToRecourseOnLabel)
        plt.xlabel('Round')
        plt.ylabel('Fail_to_Recourse_on_Label')
        plt.title('FtR during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'Fail_to_Recourse_on_Label.png'))

    #紀錄新增進來的sample資料
    def addEFTDataFrame(self,index):
        if self.EFTdataframe.at[0,'updateRounds'] != 0:
            sampleDataframe =  pd.DataFrame(
                {
                    'x':self.train.x[index].tolist(),
                    'y':self.train.y[index].flatten(),
                    # 'Predict':np.nan,
                    # 'Predict':self.train.y[index].flatten(),
                    'Predict':[ [] for _ in range(len(train.y[index]))],
                    'flip_times':np.zeros(len(self.train.y[index])),
                    'startRounds':np.full(len(self.train.y[index]),int(self.EFTdataframe.at[0,'updateRounds'])),
                    'updateRounds':np.zeros(len(self.train.y[index])),
                    'EFT' : np.zeros(len(self.train.y[index])),
                    # 'EFPList': [np.zeros(int(self.EFTdataframe.at[0,'rounds'])) for _ in range(len(train.y[index]))]
                    'EFTList': [ [0.0] * int(self.EFTdataframe.at[0,'updateRounds']) for _ in range(len(self.train.y[index]))]
                }
            )
        else:
            sampleDataframe =  pd.DataFrame(
                {
                    'x':self.train.x[index].tolist(),
                    'y':self.train.y[index].flatten(),
                    # 'Predict':np.nan,
                    # 'Predict':self.train.y[index].flatten(),
                    'Predict':[ [] for _ in range(len(train.y[index]))],
                    'flip_times':np.zeros(len(self.train.y[index])),
                    'startRounds':np.zeros(len(self.train.y[index])),
                    'updateRounds':np.zeros(len(self.train.y[index])),
                    'EFT' : np.zeros(len(self.train.y[index])),
                    'EFTList': [ [] for _ in range(len(self.train.y[index]))]
                }
            )
        # display(sampleDataframe)
        self.EFTdataframe = pd.concat([self.EFTdataframe,sampleDataframe],ignore_index=True)

    def calculate_accuracy(self, predicted_results, actual_labels, threshold=0.5):
      # Convert probabilities to binary predictions based on the threshold
      binary_predictions = []
      for prob in predicted_results:
        pred_label = 0
        if prob >= threshold:
          pred_label = 1
        else:
          pred_label = 0
        binary_predictions.append(pred_label)

      # Compare binary predictions to actual labels
      correct_predictions = []
      for i in range(0, len(binary_predictions)):
        if(binary_predictions[i] == actual_labels[i]):
          correct_predictions.append(1)
        else:
          correct_predictions.append(0)

      # Calculate accuracy as the ratio of correct predictions to total predictions
      if len(correct_predictions) == 0:
        return 0
      accuracy = sum(correct_predictions) / len(correct_predictions)
      return accuracy

    def calculate_AA(self, kth_model: nn.Module, jth_data_after_recourse: list):
      if jth_data_after_recourse:
        kth_model.eval()
        sum = 0

        # do each historical task
        for j in jth_data_after_recourse:
          pred = kth_model(j.x)
          acc = self.calculate_accuracy(pred, j.y)
          sum += acc

        return sum / len(jth_data_after_recourse)

      print("jth_data_after_recourse cannot be empty")
      return None

    def calculate_BWT(self, kth_model: nn.Module, jth_data_after_recourse, Ajj_performance_list):
      kth_model.eval()
      sum = 0

      for i in range (0, len(jth_data_after_recourse)):
        # if is the last loop, calculate A(j,j) and store it
        if i == len(jth_data_after_recourse) - 1:
          pred = kth_model(jth_data_after_recourse[i].x)
          Ajj_performance_list.append(self.calculate_accuracy(pred, jth_data_after_recourse[-1].y))
        # else we calculate A(k,j) - A(j,j)
        else:
          pred = kth_model(jth_data_after_recourse[i].x)
          acc = self.calculate_accuracy(pred, jth_data_after_recourse[i].y) - Ajj_performance_list[i]
          sum += acc

      if len(jth_data_after_recourse) == 1:
        return sum
      return sum / (len(jth_data_after_recourse) - 1)

    def calculate_FWT(self, Ajj_performance_list, Aj_tide_list):
      sum = 0

      for i in range(1, len(Ajj_performance_list)):
        sum +=  Ajj_performance_list[i] - Aj_tide_list[i]
        # print(Ajj_performance_list[i], Aj_tide_list[i])
      if len(Aj_tide_list) == 1:
        return sum
      return sum / (len(Aj_tide_list) - 1)

    def plot_matricsA(self):
      # Create a figure and subplots
      fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

      # Plot data on each subplot and draw
      axs[0].plot(self.overall_acc_list)
      axs[0].set_title('avarage accuracy')
      axs[0].set_xlabel('Round')
      axs[0].set_ylabel('accuracy')

      axs[1].plot(self.memory_stability_list)
      axs[1].set_title('memory stability (BWT)')
      axs[1].set_xlabel('Round')
      axs[1].set_ylabel('memory stability')

      axs[2].plot(self.memory_plasticity_list)
      axs[2].set_title('learning plasticity')
      axs[2].set_xlabel('Round')
      axs[2].set_ylabel('learning plasticity')

      # Adjust layout
      plt.tight_layout()

      # Show plot
      plt.savefig(os.path.join(self.save_directory, 'matricsA.png'))

    def plot_Ajj(self):
      plt.figure()
      plt.plot(self.Ajj_performance_list)
      plt.xlabel('Round')
      plt.ylabel('Ajj accuracy')
      plt.title('Ajj accuracy')
      plt.savefig(os.path.join(self.save_directory, 'Ajj.png'))

    def plot_jsd(self):
      plt.figure()
      plt.plot(self.jsd_list)
      plt.xlabel('Round')
      plt.ylabel('js divergence')
      plt.title('js divergence')
      plt.savefig(os.path.join(self.save_directory, 'jsd.png'))

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        raise NotImplementedError()
    
    
