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
from Dataset.makeDataset import Dataset as makeDataset
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config.config import test, train, sample
from Models.synapticIntelligence import SynapticIntelligence

pca = PCA(2).fit(train.x)

class Helper:
    palette = sns.color_palette('muted', 2)
    cmap = ListedColormap(palette)

    def __init__(self, model: nn.Module, pca: PCA, train: Dataset, test: Dataset, sample: Dataset):
        self.model = model
        print(self.model)
        self.pca = pca
        self.train = train
        self.last_train = None
        self.test = test
        self.recoursedFail = []
        self.recoursedSuccess = []
        self.sample = sample
        self.avgRecourseCost_list = []
        self.avgRecourseCostOnNormalModel_list = [0.06159922992002644, 0.12086665188383333, 0.12428388443361009, 0.09327902214606151, 0.0914254747813962, 0.10181462205051678, 0.06512063602287499, 0.05560648227644058, 0.14264473209587378, 4.09476747603947e-10, 0.052722483241979104, 0.19565176651228425, 0.06654292898757293, 0.06011031716706189, 0.09046107084462783, 0.09270152840961687, 0.051453308538206, 0.03577024625382576, 0.016691426807023336, 0.08029367543314575, 0.026682426600765776, 0.03204131528429175, 0.04240766617553736, 0.014405035321681572, 0.04639084834714248, 0.050464024788218174, 0.012282079520451657, 0.05285377481895858, 0.059932515457909896, 0.04251443191028677, 0.05739795532695681, 0.05558917593451114, 0.028256987581770848, 6.73931037377352e-25, 0.09796421873509148, 0.03888524785536303, 0.04657775858007727, 0.05730771480139618, 1.5402928308309566e-06, 0.06509942725759371, 0.0491878879805011, 0.00039157206276569187, 0.07065706911599993, 0.06210125794068486, 0.04395887542651051, 0.043938844512779185, 0.02238094295522375, 0.062207699089470395, 0.08282516573412534, 0.09728769390268684, 0.044728735811752546, 0.037621906280156193, 0.05039526779801566, 0.050188176154787205, 0.024204884508321714, 3.260193750713085e-20, 0.09291165744579091, 0.02995874151072171, 0.008952477908322693, 0.052271700241058636, 8.04082868080906e-05, 0.03617987324863985, 0.0017337123456435944, 0.017247563861170864, 0.01080249299107777, 0.048440334502938, 1.4813621746908996e-16, 0.03357194517838013, 0.033571785361655185, 0.02210148653736829, 0.03650270177876014, 0.05248624145473789, 0.010576933945023938, 0.0670926706085056, 1.4454213688446538e-05, 0.02168657426886433, 0.02674180725114156, 0.029348920762914146, 0.07503762732903632]
        self.ratioOfDifferentLabel = []
        self.fairRatio = []
        self.q3RecourseCost = []
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
        self.failToRecourseBeforeModelUpdate = []
        self.failToRecourseOnNormalModel = [0.0, 0.053763440860215055, 0.12903225806451613, 0.030927835051546393, 0.0425531914893617, 0.04081632653061224, 0.0, 0.0, 0.07291666666666667, 1.0, 0.0, 0.0, 0.0, 0.0, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.011494252873563218, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.010416666666666666, 0.011235955056179775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02040816326530612, 0.0, 0.0, 0.010752688172043012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02127659574468085, 0.0, 0.0, 0.0, 0.0196078431372549, 0.010526315789473684, 0.0, 0.009708737864077669, 0.0, 0.0, 0.010752688172043012, 0.0, 0.0, 0.0, 0.009900990099009901, 0.024096385542168676, 0.0125, 0.0, 0.0425531914893617, 0.0, 0.0, 0.0, 0.0, 0.02040816326530612, 0.0, 0.0, 0.0, 0.0, 0.019230769230769232, 0.020833333333333332]
        self.recourseModelLossList = []
        self.RegreesionModelLossList = []
        self.RegreesionModel_valLossList = []

        self.validation_list = []
        self.Ajj_performance_list = []
        self.overall_acc_list = []
        self.memory_stability_list = []
        self.memory_plasticity_list = []
        self.Aj_tide_list = []
        self.jsd_list = []

        self._hist_last: list[BarContainer]
        self._bins_last: NDArray
        self._hist_current: list[BarContainer]
        self._bins_current: NDArray
        self._sc_train: PathCollection
        self._sc_test: PathCollection
        self._sc_recourse_fail: PathCollection
        self._sc_recourse_success: PathCollection
        self._ct_test: QuadContourSet
        self._ct_train: QuadContourSet
        self.lr = 0.1
        self.si: SynapticIntelligence
        self.save_directory = None

        self.testacc = []
        self.cnt = 0
        self.avgNewRecourseCostList = []
        self.avgOriginalRecourseCostList = []
        self.historyTrainList = []
        self.train_size = 0
        self.t_rate_list = []
        self.model_params = None
        self.first_model_params = None
        self.model_shift_distance_list = []
        self.failToRecourse_old = []
        self.failToRecourse_new = [] 
        self.showRecoursedPoints = True
        self.low_cost_model_shift_list = []
        self.high_cost_model_shift_list = []
        self.low_cost_feature_ranking = []
        self.important_feature_ranking = []

    # def draw_proba_hist(self, ax: Axes | None = None, *, label: bool = False):
    def draw_proba_hist(self, ax0: Axes = None, ax1: Axes = None, *, label: bool = False):
        if ax0 is None:
            fig0, ax0 = plt.subplots(figsize=(4, 4))
        else:
            fig0 = ax0.get_figure()
        if ax1 is None:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
        else:
            fig1 = ax1.get_figure()

        # For ax0, using self.last_train
        if hasattr(self, 'last_train') and self.last_train is not None:
            x_last = self.last_train.x
            y_last = self.last_train.y
        else:
            x_last = self.train.x
            y_last = self.train.y

        n_last = x_last.shape[0] #size of last_train data
        m_last = n_last - pt.count_nonzero(y_last) #non zero size of last_train data 
        
        with pt.no_grad():
            y_prob_last: pt.Tensor = self.model(x_last)

        # Sorts the probabilities based on the order of the labels (groups by class)
        y_last = y_last.flatten()
        y_prob_last = y_prob_last.flatten()
        y_prob_last = y_prob_last[y_last.argsort()] 

        # Creates weights for the histogram - each sample contributes a percentage to make totals sum to 100%
        w_last = np.broadcast_to(100 / n_last, n_last)

        _, self._bins_last, self._hist_last = ax0.hist(
            (y_prob_last[:m_last], y_prob_last[m_last:]),
            10,
            (0, 1),
            weights=(w_last[:m_last], w_last[m_last:]),
            rwidth=1,
            color=self.palette,
            label=(0, 1),
            ec='w',
            alpha=0.9,
        )

        ax0.legend(loc='upper center', title='Topk label')
        ax0.set_xlabel('predicted probability')
        ax0.set_ylabel('percentage')
        ax0.set_title('User Responded Dataset on Model t - 1')

        if label:
            for c in self._hist_last:
                height = map(Rectangle.get_height, c.patches)
                ax0.bar_label(
                    c,
                    [f'{h}%' if h else '' for h in height],
                    fontsize='xx-small'
                )
        ax0.set_ylim(0, 80)

        # For ax1, using self.train
        x = self.train.x
        y = self.train.y

        n = x.shape[0]
        m = n - pt.count_nonzero(y)

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x)

        y = y.flatten()
        y_prob = y_prob.flatten()
        y_prob = y_prob[y.argsort()]

        w = np.broadcast_to(100 / n, n)

        _, self._bins_current, self._hist_current = ax1.hist(
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

        ax1.legend(loc='upper center', title='Topk label')
        ax1.set_xlabel('predicted probability')
        ax1.set_title('User Responded Dataset on Model t')

        if label:
            for c in self._hist_current:
                height = map(Rectangle.get_height, c.patches)
                ax1.bar_label(
                    c,
                    [f'{h}%' if h else '' for h in height],
                    fontsize='xx-small'
                )
        ax1.set_ylim(0, 80)
        
        return (fig0, ax0), (fig1, ax1)

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
        # print("js_divergence: ",js_divergence)
        self.jsd_list.append(js_divergence)

    # def draw_dataset_scatter(self, axes: tuple[Axes, Axes] | None = None):
    def draw_dataset_scatter(self, axes: [Axes, Axes] = None):
        if axes is None:
            # Increase the figure size for larger plotting area
            fig, (ax0, ax1) = plt.subplots(
                1, 2,
                sharex=True,
                sharey=True,
                figsize=(8, 4),
                layout='constrained'
            )
        else:
            ax0, ax1 = axes
            fig = ax0.get_figure()

        prop = dict(cmap=self.cmap, s=40, vmin=0., vmax=1., lw=0.8, ec='w')
        
        # Standard scatter plot for training data
        self._sc_train = ax0.scatter(
            *pca.transform(self.train.x).T,
            c=self.train.y,
            **prop
        )
        
        if(self.showRecoursedPoints):
            # Extract the recoursedFail points from training data
            if hasattr(self, 'recoursedFail') and len(self.recoursedFail) > 0:
                # Convert recoursedFail to tensor indices if it's a list
                if isinstance(self.recoursedFail, list):
                    recourse_fail_indices = pt.tensor(self.recoursedFail, dtype=pt.long)
                else:
                    recourse_fail_indices = self.recoursedFail
                    
                # Get the PCA-transformed coordinates of recourse points
                recourse_fail_points = pca.transform(self.train.x[recourse_fail_indices])
                
                # Plot the recoursedFail points in green over the original scatter plot
                self._sc_recourse_fail = ax0.scatter(
                    *recourse_fail_points.T,
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='purple',      
                )
                
            else:
                # Initialize with empty data so the attribute exists
                self._sc_recourse_fail = ax0.scatter(
                    [], 
                    [], 
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='purple',
                )
            
            # Extract the recoursedSuccess points from training data
            if hasattr(self, 'recoursedSuccess') and len(self.recoursedSuccess) > 0:
                # Convert recoursedSuccess to tensor indices if it's a list
                if isinstance(self.recoursedSuccess, list):
                    recourse_success_indices = pt.tensor(self.recoursedSuccess, dtype=pt.long)
                else:
                    recourse_success_indices = self.recoursedSuccess
                    
                # Get the PCA-transformed coordinates of recoursedSuccess points
                recourse_success_points = pca.transform(self.train.x[recourse_success_indices])
                
                # Plot the recoursedFail points in green over the original scatter plot
                self._sc_recourse_success = ax0.scatter(
                    *recourse_success_points.T,
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='red',      
                )
                
            else:
                # Initialize with empty data so the attribute exists
                self._sc_recourse_success = ax0.scatter(
                    [], 
                    [], 
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='red',
                )

        if(self.showRecoursedPoints):
            handles = []
            labels = []
            # Add training scatter points to legend (if needed)
            if hasattr(self, '_sc_train'):
                train_handles, train_labels = self._sc_train.legend_elements()
                handles.extend(train_handles)
                labels.extend([label for label in train_labels])

            # Add recourse success points to legend
            if hasattr(self, '_sc_recourse_success'):
                handles.append(self._sc_recourse_success)
                labels.append('R_1')

            # Add recourse fail points to legend
            if hasattr(self, '_sc_recourse_fail'):
                handles.append(self._sc_recourse_fail)
                labels.append('R_0')

            # Create the combined legend
            ax0.legend(handles, labels, loc='upper right', title='topk label')
        
        else:
            ax0.legend(
                *self._sc_train.legend_elements(),
                loc='upper right',
                title='topk label'
            )

        # Add PCA variance to labels
        ax0.set_xlabel(f'PCA1')
        ax0.set_ylabel(f'PCA2')
        ax0.grid(alpha=0.75)
        ax0.set_title('User Responded Dataset at time t')
        
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
            title='model label'
        )

        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        
        # Increase the expand factor for larger PCA area
        expand_factor = 1.0
        x_range = (x1 - x0) * expand_factor
        y_range = (y1 - y0) * expand_factor

        # Set new limits
        ax0.set_xlim([x0 - x_range, x1 + x_range])
        ax0.set_ylim([y0 - y_range, y1 + y_range])
        ax1.set_xlim([x0 - x_range, x1 + x_range])
        ax1.set_ylim([y0 - y_range, y1 + y_range])
        
        # Increase grid resolution for smoother contours
        n = 32
        x_expanded = np.linspace(x0 - x_range, x1 + x_range, n)
        y_expanded = np.linspace(y0 - y_range, y1 + y_range, n)

        xy = np.meshgrid(x_expanded, y_expanded)

        z = pca.inverse_transform(np.c_[xy[0].ravel(), xy[1].ravel()])
        z = pt.tensor(z, dtype=pt.float)

        # Evaluate the model on the expanded grid
        with pt.no_grad():
            z = self.model(z)
        z = z.view(n, n)
        
        self._ct_test = ax1.contourf(
            *xy, z, 10,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            alpha=0.9,
            zorder=0,
        )
        
        # Also add the contour to the first plot
        self._ct_train = ax0.contourf(
            *xy, z, 10,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            alpha=0.9,
            zorder=0,
        )
        
        fig.colorbar(self._ct_test, ax=ax1, label='probability')
        ax1.grid(alpha=0.75) 
        
        # Also update the x-label for the second plot
        ax1.set_xlabel(f'PCA1')
        ax1.set_title('Initial Distribution Dataset')

        return fig, axes

    def draw_all(self):
        sf: list[SubFigure]
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        sf = fig.subfigures(2, 2)
        ax0 = sf[0, 0].subplots()
        ax1 = sf[0, 1].subplots()
        bottom_sf = fig.subfigures(2, 1)[1]
        ax2, ax3 = bottom_sf.subplots(1, 2, sharex=True, sharey=True)
        self.draw_proba_hist(ax0, ax1)
        self.draw_dataset_scatter((ax2, ax3))
        return fig, (ax0, ax1, ax2, ax3)

    def animate_all(self, frames: int = 120, fps: int = 10, *, inplace: bool = False):
        fig, (ax0, ax1, ax2, ax3) = self.draw_all()

        # print("inplace: ",inplace)
        # model = self.model if inplace else deepcopy(self.model)
        # train = self.train if inplace else deepcopy(self.train)
        # model = self.model
        # train = self.train
        # test = self.test
        # sample = self.sample

        def init():
            return *ax0.patches, *ax1.patches, *ax2.collections, *ax3.collections

        def func(frame):
            # PCA for training data
            pca_train = PCA(2).fit(train.x)

            # Separate PCA for test data
            pca_test = PCA(2).fit(test.x)

            fig.suptitle(f't = {frame}', ha='left', x=0.01, size='small')

            if frame == 0:
                return ()

            # Update histograms for last_train (ax0)
            # store the last training data here before update the train data in self.update()
            self.last_train = makeDataset(self.train.x.clone(), self.train.y.clone())
            # compute the probibility of last training data on last model before update
            with pt.no_grad():
                if(self.last_train is not None):
                    y_prob_last: pt.Tensor = self.model(self.last_train.x)
                    y_last = self.last_train.y.flatten()
                    n_last = self.last_train.x.shape[0]
                else:
                    y_prob_last = None

            self.update(self.model, self.train, self.sample, self.recoursedFail, self.recoursedSuccess)

            # Update histograms for last_train (ax0)
            # here we handle the case that in first 2 rounds there is no recourse,
            # so there is no modification on train data and model. We set last train data = current train data
            with pt.no_grad():
                if(self.last_train is None):
                    y_prob_last: pt.Tensor = self.model(self.train.x)
                    y_last = self.train.y.flatten()
                    n_last = self.train.x.shape[0]     

            y_prob_last = y_prob_last.flatten()
            m_last = n_last - pt.count_nonzero(y_last)
            rank_last = y_prob_last[y_last.argsort()]

            # Update histogram for last train data (ax0)
            for b, r in zip(self._hist_last, (rank_last[:m_last], rank_last[m_last:])):
                height, _ = np.histogram(r, self._bins_last, range=(0, 1))
                for rect, h in zip(b.patches, height * (100 / n_last)):
                    rect.set_height(h)

            # Update histograms for current train (ax1)
            with pt.no_grad():
                y_prob_current: pt.Tensor = self.model(self.train.x)

            y_current = self.train.y.flatten()
            y_prob_current = y_prob_current.flatten()
            n_current = self.train.x.shape[0]
            m_current = n_current - pt.count_nonzero(y_current)
            rank_current = y_prob_current[y_current.argsort()]

            # Update histogram for current train data (ax1)
            for b, r in zip(self._hist_current, (rank_current[:m_current], rank_current[m_current:])):
                height, _ = np.histogram(r, self._bins_current, range=(0, 1))
                for rect, h in zip(b.patches, height * (100 / n_current)):
                    rect.set_height(h)

            # Use pca_train for training data
            self._sc_train.set_offsets(pca_train.transform(train.x))
            self._sc_train.set_array(train.y.flatten())

            if(self.showRecoursedPoints):
                if hasattr(self, 'recoursedFail') and len(self.recoursedFail) > 0:
                    # Get the indexed array
                    indexed_array = self.train.x[self.recoursedFail]
                    
                    # Check if the indexed array actually has samples
                    if indexed_array.shape[0] > 0:
                        self._sc_recourse_fail.set_offsets(pca_train.transform(indexed_array))
                        
                if hasattr(self, 'recoursedSuccess') and len(self.recoursedSuccess) > 0:
                    # Get the indexed array
                    indexed_array = self.train.x[self.recoursedSuccess]
                    
                    # Check if the indexed array actually has samples
                    if indexed_array.shape[0] > 0:
                        self._sc_recourse_success.set_offsets(pca_train.transform(indexed_array))

            # calculate js divergence of pca training data
            self.js_divergence(self._sc_train.get_offsets(), self._sc_train.get_array())
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(test.x)

            y_prob = y_prob.flatten()
            y_pred = y_prob.greater(0.5)

            # Use pca_test for test data
            self._sc_test.set_offsets(pca_test.transform(test.x))
            self._sc_test.set_array(y_pred)


            ax2.relim()
            ax2.autoscale_view()

            ax3.relim()
            ax3.autoscale_view()

            for c in self._ct_test.collections:
                c.remove()

            for c in self._ct_train.collections:
                c.remove()

            # For training plot (ax2)
            x0_train, x1_train = ax2.get_xlim()
            y0_train, y1_train = ax2.get_ylim()
            n = 32
            xy_train = np.mgrid[x0_train: x1_train: n * 1j, y0_train: y1_train: n * 1j]
            z_train = pca_train.inverse_transform(xy_train.reshape(2, n * n).T)
            z_train = pt.tensor(z_train, dtype=pt.float)
            with pt.no_grad():
                z_train: pt.Tensor = self.model(z_train)
            z_train = z_train.view(n, n)

            self._ct_train: QuadContourSet = ax2.contourf(
                *xy_train, z_train, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )

            # For test plot (ax3)
            x0_test, x1_test = ax3.get_xlim()
            y0_test, y1_test = ax3.get_ylim()
            xy_test = np.mgrid[x0_test: x1_test: n * 1j, y0_test: y1_test: n * 1j]
            z_test = pca_test.inverse_transform(xy_test.reshape(2, n * n).T)
            z_test = pt.tensor(z_test, dtype=pt.float)
            with pt.no_grad():
                z_test: pt.Tensor = self.model(z_test)
            z_test = z_test.view(n, n)

            self._ct_test: QuadContourSet = ax3.contourf(
                *xy_test, z_test, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )

            n_recourse_success = len(self.recoursedSuccess) if hasattr(self, 'recoursedSuccess') else 0
            n_recourse_fail = len(self.recoursedFail) if hasattr(self, 'recoursedFail') else 0
            sc_train_labels = self._sc_train.get_array()  # this holds labels for plotted train points

            # n_label_1 = np.sum(sc_train_labels == 1)
            # n_label_0 = np.sum(sc_train_labels == 0)
            # print("n_label_1: ",n_label_1)
            # print("n_label_0: ",n_label_0)
            # print("n_recourse_success: ",n_recourse_success)
            # print("n_recourse_fail: ",n_recourse_fail)

            return *ax0.patches, *ax1.patches, *ax2.collections, *ax3.collections

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
        
    def draw_avgRecourseCostCompareToNormalModel(self):
        x = [20] * 20 + [40] * 20 + [60] * 20 + [80] * 19
        df_recourseFailRate = pd.DataFrame({
            'rounds': x,
            'avgRecourseCost': self.avgRecourseCost_list,
            'model' : ['now'] * 79
        })
        df_recourseFailRateNormalModel = pd.DataFrame({
            'rounds': x,
            'avgRecourseCost': self.avgRecourseCostOnNormalModel_list,
            'model' : ['normal'] * 79
        })
        df_combined = pd.concat([df_recourseFailRate, df_recourseFailRateNormalModel], ignore_index=True)
        plt.figure()
        # sns.boxplot(x='rounds',y='EFPList',data = self.EFTdataframe)
        sns_plot = sns.boxplot(x = 'rounds',y = 'avgRecourseCost',hue= 'model',data = df_combined)
        plt.savefig(os.path.join(self.save_directory, 'avgRecourseCostCompareToNormalModel.png'))

    def draw_failToRecourseCompareToNormalModel(self):
        # recourseFailRate = pd.Series(self.failToRecourse)
        # x = pd.Series([20,40,60,80])
        x = [20] * 20 + [40] * 20 + [60] * 20 + [80] * 19
        df_recourseFailRate = pd.DataFrame({
            'rounds': x,
            'failRate': self.failToRecourse,
            'model' : ['now'] * 79
        })
        df_recourseFailRateNormalModel = pd.DataFrame({
            'rounds': x,
            'failRate': self.failToRecourseOnNormalModel,
            'model' : ['normal'] * 79
        })
        df_combined = pd.concat([df_recourseFailRate, df_recourseFailRateNormalModel], ignore_index=True)
        # print("self.failToRecourse",self.failToRecourse)
        plt.figure()
        # sns.boxplot(x='rounds',y='EFPList',data = self.EFTdataframe)
        sns_plot = sns.boxplot(x = 'rounds',y = 'failRate',hue= 'model',data = df_combined)
        plt.savefig(os.path.join(self.save_directory, 'failToRecourseCompareToNormalModel.png'))
        # sns_plot = sns.boxplot(x = 'rounds',y = 'failRate',data = df_recourseFailRate)
        # fig = sns_plot.get_figure()
        # fig.savefig("failToRecourseCompareToNormalModel.png") 
    
    def draw_EFT(self,epochs):
        data = []
        labels = [(epochs / 10 - 1) * (i+1) for i in range(10)]
        x = [1,2,3,4,5,6,7,8,9,10]
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

    def plot_t_rate(self):
        plt.figure()
        plt.plot(self.t_rate_list)
        plt.xlabel('Round')
        plt.ylabel('t_rate')
        plt.title('t_rate during Rounds')
        plt.savefig(os.path.join(self.save_directory, 't_rate.png'))

    def plot_model_shift(self):
        plt.figure()
        plt.plot(self.model_shift_distance_list)
        plt.xlabel('Round')
        plt.ylabel('model_shift_distance')
        plt.title('model_shift_distance during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'model_shift_distance.png'))

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
        
    def draw_failToRecourseBeforeModelUpdate(self):
        plt.figure()
        plt.plot(self.failToRecourseBeforeModelUpdate)
        plt.xlabel('Round')
        plt.ylabel('Fail_to_Recourse')
        plt.title('FtR during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'Fail_to_Recourse_BeforeUpdate.png'))
    
    def draw_Fail_to_Recourse_with_Model_Loss(self):
        plt.figure()
        plt.plot(self.recourseModelLossList,label='RecourseLoss',color='red')
        plt.plot(self.failToRecourse,label='FailtoRecourse',color='green')
        plt.plot(self.failToRecourseBeforeModelUpdate,label='FailtoRecourse_Before',color='blue')
        plt.xlabel('Round')
        plt.title('FtR_RecourseLoss during Rounds')
        plt.legend()
        plt.savefig(os.path.join(self.save_directory, 'Fail_to_Recourse_with_RecourseLoss.png'))
        
    def draw_recourseModelLoss(self):
        plt.figure()
        plt.plot(self.recourseModelLossList)
        plt.xlabel('Round')
        plt.ylabel('recourseModel_Loss')
        plt.title('recourseModel_Loss')
        plt.savefig(os.path.join(self.save_directory, 'recourseModel_Loss.png'))
        
    def draw_RegressionModelLoss(self):
        plt.figure()
        plt.plot(self.RegreesionModelLossList)
        plt.xlabel('Round')
        plt.ylabel('RegreesionModel_Loss')
        plt.title('RegreesionModel_Loss')
        plt.savefig(os.path.join(self.save_directory, 'RegreesionModel_Loss.png'))
    
    def draw_RegressionModelValLoss(self):
        plt.figure()
        plt.plot(self.RegreesionModel_valLossList)
        plt.xlabel('Round')
        plt.ylabel('RegreesionModel_valLoss')
        plt.title('RegreesionModel_valLoss')
        plt.savefig(os.path.join(self.save_directory, 'RegreesionModel_valLoss.png'))
        

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
    
    def _weight_func(self, x):
        return 1
    
    def _get_weight(self, observe_range):
        weights = []
        for i in range (1, observe_range + 1):
            weights.append(self._weight_func(i))

        sum = 0
        for i in weights:
            sum += i
        return [w / sum for w in weights]

    def calculate_AA(self, kth_model: nn.Module, jth_data_after_recourse: list, rangenum):
        rangenum = min(rangenum, len(jth_data_after_recourse))
        rangenum -= 1
        if jth_data_after_recourse:
            kth_model.eval()
            sum = 0
            weights = self._get_weight(rangenum)

            
            # do each historical task
            for j in range(-2, -rangenum - 2, -1):
                pred = kth_model(jth_data_after_recourse[j].x)
                acc = self.calculate_accuracy(pred, jth_data_after_recourse[j].y) * weights[j+1]
                self.testacc.append(acc)
                sum += acc

            self.testacc.append('|')
            return sum

        print("jth_data_after_recourse cannot be empty")
        return None

    def calculate_BWT(self, kth_model: nn.Module, jth_data_after_recourse, Ajj_performance_list, rangenum):
        kth_model.eval()
        sum = 0
        weights = self._get_weight(rangenum - 1)

        for i in range (0, len(jth_data_after_recourse)):
            # if is the last loop, calculate A(j,j) and store it
            if i == len(jth_data_after_recourse) - 1:
                pred = kth_model(jth_data_after_recourse[i].x)
                Ajj_performance_list.append(self.calculate_accuracy(pred, jth_data_after_recourse[-1].y))
            # else we calculate A(k,j) - A(j,j)
            else:
                if(len(jth_data_after_recourse) - rangenum <= i and i < len(jth_data_after_recourse) - 1):
                    pred = kth_model(jth_data_after_recourse[i].x)
                    acc = self.calculate_accuracy(pred, jth_data_after_recourse[i].y) - Ajj_performance_list[i]
                    acc = acc * weights[i - len(jth_data_after_recourse) + rangenum]
                    sum += acc  

        if len(jth_data_after_recourse) < rangenum:
            return 0
        
        return sum

    def calculate_FWT(self, Ajj_performance_list, Aj_tide_list, rangenum):
      sum = 0
      if(len(Aj_tide_list) < rangenum):
          return 0
      
      weights = self._get_weight(rangenum - 1)
      for i in range(-1, -rangenum, -1):
        sum +=  (Ajj_performance_list[i] - Aj_tide_list[i]) * weights[i]
        # print(Ajj_performance_list[i], Aj_tide_list[i])
      
      return sum

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

    def plot_aac(self):
        plt.figure()
        x_values = range(2, 2 + len(self.overall_acc_list))
        plt.plot(x_values, self.overall_acc_list)
        plt.xlabel('Round')
        plt.ylabel('Short-term Accuracy')
        plt.title('Short-term Accuracy')
        plt.savefig(os.path.join(self.save_directory, 'aac.png'))
        plt.close()
      
    def draw_avgRecourseCost(self):
        # Check if there is any data to plot
        if any(len(lst) > 0 for lst in [self.avgRecourseCost_list, self.avgNewRecourseCostList, self.avgOriginalRecourseCostList]):
            plt.figure()
    
            if len(self.avgRecourseCost_list) > 0:
                plt.plot(self.avgRecourseCost_list, label='Average Recourse Cost')
            if len(self.avgNewRecourseCostList) > 0:
                plt.plot(self.avgNewRecourseCostList, label='Average New Recourse Cost', linestyle='--')
            if len(self.avgOriginalRecourseCostList) > 0:
                plt.plot(self.avgOriginalRecourseCostList, label='Average Original Recourse Cost', linestyle='-.')
            
            # Add labels, title, legend, and save the figure
            plt.xlabel('Round')
            plt.ylabel('Cost')
            plt.title('Recourse Cost Comparison')
            plt.legend()
            plt.savefig(os.path.join(self.save_directory, 'avgRecourseCostComparison.png'))
            plt.close()
            
    def draw_testDataFairRatio(self):
        plt.figure()
        plt.plot(self.fairRatio)
        plt.xlabel('Round')
        plt.ylabel('fair_ratio')
        plt.title('testDataLabelConsistency')
        plt.savefig(os.path.join(self.save_directory, 'testDataLabelConsistency.png'))
        
    def draw_q3RecourseCost(self):
        plt.figure()
        plt.plot(self.q3RecourseCost)
        plt.xlabel('Round')
        plt.ylabel('Recourse_Cost')
        plt.title('Q3_Recourse_Cost')
        plt.savefig(os.path.join(self.save_directory, 'Q3RecourseCost.png'))
        
    def draw_topkRatioOfDifferentLabel(self):
        # y_prob = deepcopy(y_prob_all)
        # print("y_prob: ",y_prob)
        # print("y_prob_all: ",y_prob_all)
        #do the labeling according to the probability
        # y_prob[y_prob > 0.5] = 1
        # y_prob[y_prob <= 0.5] = 0
        # print("After labeling")
        # print("y_prob: ",y_prob)
        # print("y_prob_all: ",y_prob_all)
        plt.figure()
        plt.plot(self.ratioOfDifferentLabel)
        plt.xlabel('Round')
        plt.ylabel('RatioOfDifferentLabel')
        plt.title('RatioOfDifferentLabel')
        plt.savefig(os.path.join(self.save_directory, 'RatioOfDifferentLabel.png'))

    def plot_jsd(self):
      plt.figure()
      plt.plot(self.jsd_list)
      plt.xlabel('Round')
      plt.ylabel('js divergence')
      plt.title('js divergence')
      plt.savefig(os.path.join(self.save_directory, 'jsd.png'))

    def update(self, model: nn.Module, train: Dataset, sample: Dataset):
        raise NotImplementedError()
    
    
    def findDiverse(self, points, k):
        if k == 1:
            return [0]
        if k >= len(points):
            return list(range(len(points)))

        # generate the distance matrix
        differences = points.unsqueeze(1) - points.unsqueeze(0)
        distances = pt.norm(differences, dim=2)

        chosen = []
        chosen.append(0)
        nums = len(points)
        
        # find k disperse point
        for _ in tqdm(range(1, k), desc="Progress"):
            pth_min = 0
            choose = 1
            
            # find the point that has the largest minimum distance to the chosen points
            for i in range(1, nums):
                if i in chosen:
                    continue
                min = distances[i][chosen[0]]
                for j in chosen:
                    if i == j:
                        continue
                    if distances[i][j] < min:
                        min = distances[i][j]
                if min > pth_min:
                    pth_min = min
                    choose = i
            
            chosen.append(choose)

        return chosen
    
    def plot_model_shift_w_diff_cost(self):
        plt.figure()
        plt.plot(self.low_cost_model_shift_list)
        plt.plot(self.high_cost_model_shift_list)
        plt.legend(['low_cost_model_shift', 'high_cost_model_shift'])
        plt.xlabel('Round')
        plt.ylabel('model_shift_distance')
        plt.title('model_shift_distance during Rounds')
        plt.savefig(os.path.join(self.save_directory, 'model_shift_comp.png'))

    
    def plot_feature_ranking(self):
        plt.figure()
        plt.plot(self.important_feature_ranking)
        plt.plot(self.low_cost_feature_ranking)
        plt.legend(["important feature", "low cost feature"])
        plt.xlabel('Round')
        plt.ylabel('feature ranking')
        plt.title('feature ranking during rounds')
        plt.savefig(os.path.join(self.save_directory, 'feature ranking.png'))
            
                
    
