import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from pyfr.observer import IntegratorPerformanceObserver
from pyfr.mpiutil import get_comm_rank_root

class Modeller:
    def __init__(self, intg):
        self.intg = intg

    def set_bounds(self, bounds):
        bounds_array = np.array(bounds)
        self.bound_lower = bounds_array[:, 0]
        self.bound_upper = bounds_array[:, 1]

    def set_preprocessing(self):
        if not hasattr(self, 'y_transform'):
            self.y_transform = StandardScaler()

            # log of x is a Gaussian distribution. So we should first transform x to log(x) to make it Gaussian and then apply MinMaxScaler
            self.x_transform = MinMaxScaler()
            self.x_transform.fit_transform = lambda x: self.x_transform.fit(np.log(x)).transform(np.log(x))
            self.x_transform.inverse_transform = lambda x: np.exp(self.x_transform.inverse_transform(x))
            self.x_transform.transform = lambda x: self.x_transform.transform(np.log(x))

    @property
    def x(self) -> np.ndarray:
        return self.x_transform.inverse_transform(self._x_data)

    @x.setter
    def x(self, value: np.ndarray):
        self._x_data = self.x_transform.fit_transform(value)

    @property
    def y(self) -> np.ndarray:
        return self.y_transform.inverse_transform(self._y_data)
    
    @y.setter
    def y(self, value: np.ndarray):
        self._y_data = self.y_transform.fit_transform(value)

    def append_x(self, x: np.ndarray):
        
        x = np.array(x).reshape(1, -1)
        
        if not hasattr(self, '_x_data'):
            self._x_data = self.x_transform.fit_transform(x)
        else:
            old_x = self.x_transform.inverse_transform(self._x_data)
            self._x_data = self.x_transform.fit_transform(np.vstack([old_x, x]))

    def append_y_ystd(self, y: np.ndarray, ystd: np.ndarray):
        y = np.array(y).reshape(1, -1)
        ystd = np.array(ystd).reshape(1, -1)

        if not hasattr(self, '_y_data'):
            self._y_data = self.y_transform.fit_transform(y)
            self._ystd_data = self.transform_ystd(ystd)
        else:
            old_y = self.y_transform.inverse_transform(self._y_data)
            old_ystd = self.y_transform.inverse_transform(self._ystd_data)
            self._y_data = self.y_transform.fit_transform(np.vstack([old_y, y]))
            self._ystd_data = self.transform_ystd(np.vstack([old_ystd, ystd]))

    def transform_ystd(self, ystd: np.ndarray):
        if isinstance(self.y_transform, StandardScaler):
            scale = self.y_transform.scale_[0]
            ystd_transformed = ystd / scale
        else:
            raise NotImplementedError('Only StandardScaler supported for ystd')
        return ystd_transformed

class PerformanceModellerMixin(Modeller):
    @property
    def cost(self) -> float:
        return self.simulation_performance

class LoadingModellerMixin(Modeller):
    @property
    def parameters(self) -> list[int]:
        comm, rank, root = get_comm_rank_root()

        return np.array(comm.allgather(self.Nₑ), dtype=np.int32)

class GPModeller(Modeller):
    def __init__(self, intg):
        super().__init__(intg)

    def append_data(self):
        self.append_x(self.parameters[0])
        self.append_y_ystd(*self.cost)

        # Prune data if multiple data points with same x, ensure max across y is taken
        #_, unique_idx = np.unique(self._x_data, axis=0, return_index=True)
        #self._x_data = self._x_data[unique_idx]
        #self._y_data = self._y_data[unique_idx]

        comm, rank, root = get_comm_rank_root()
        
        if rank == root:
        
            print(self.x_transform.inverse_transform(self._x_data), 
                  self.y_transform.inverse_transform(self._y_data), 
                  self.y_transform.inverse_transform(self._ystd_data),
                  flush=True)

    def fit_gp(self):

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-1)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(self._x_data, self._y_data)

#        skip_first = 1
#        cv_splits = 5

        # Skip first data point
#        if self._x_data.shape[0] < cv_splits + skip_first: 
#            self.gp.fit(self._x_data, self._y_data)
#            return        

#        param_grid = {'kernel__k1__length_scale': np.logspace( -4,  0, 10),
#                      'kernel__k2__noise_level':  np.logspace(-10, -1, 10)
#                     }
#
#        grid_search = GridSearchCV(self.gp, param_grid, cv=cv_splits, 
#                                   scoring='neg_mean_squared_error', n_jobs = -1)
#
#        with warnings.catch_warnings(record=True) as w:
#            warnings.simplefilter("ignore")
#            grid_search.fit(self._x_data[skip_first:], 
#                            self._y_data[skip_first:])
#        
#        # If warnings caught, fit with latest best GP model
#        if len(w) > 0:
#            print('Grid search failed. Fitting with latest best GP model.',
#                flush = True)
#        else:
#            self.gp = grid_search.best_estimator_
#
#        self.gp.fit(self._x_data[skip_first:], self._y_data[skip_first:])
#        print('Best parameters found:', grid_search.best_params_)

    def predict_gp(self, x):
        mean, std = self.gp.predict(self.x_transform.transform(x), return_std=True)
        return self.y_transform.inverse_transform(mean.reshape(-1, 1)), \
               self.y_transform.inverse_transform( std.reshape(-1, 1))

class IntegratorPerformanceLoadingGPModeller(IntegratorPerformanceObserver, 
                                             LoadingModellerMixin,
                                             PerformanceModellerMixin,
                                             GPModeller
                                             ):
    def __init__(self, intg):
        super().__init__(intg)
        
    def view_data(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # Predict for 100 points from lower to upper bounds
        x = np.linspace(self.bound_lower[0], self.bound_upper[0], 1000).reshape(-1, 1)
        y_m, y_s = self.predict_gp(x)

        ax.plot(x, y_m, label='Predicted mean')
        ax.fill_between(  x.flatten(), 
                        y_m.flatten() - y_s.flatten(), 
                        y_m.flatten() + y_s.flatten(), 
                        alpha=0.2, label='Confidence')

        ax.scatter(self.x, self.y, label='Observed', color='orange')
        ax.set_xlabel('Number of elements') ; ax.set_ylabel('Compute time')
        ax.legend() ; ax.grid(True)
        fig.savefig('gp.png')
        plt.close(fig)
