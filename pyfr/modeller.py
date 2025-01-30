import matplotlib.pyplot as plt
import numpy as np

from pyfr.observer import IntegratorPerformanceObserver
from pyfr.mpiutil import get_comm_rank_root

class Modeller:
    def __init__(self, intg):
        self.intg = intg

    def set_scalers(self, x_scaler=None, y_scaler=None):
        """Set or compute scalers for input and output data."""
        if x_scaler is None:
            # Compute mean and std for x
            self.x_mean = np.mean(self._x_data, axis=0)
            self.x_std = np.std(self._x_data, axis=0)
        else:
            self.x_mean, self.x_std = x_scaler

        if y_scaler is None:
            # Compute mean and std for y
            self.y_mean = np.mean(self._y_data)
            self.y_std = np.std(self._y_data)
        else:
            self.y_mean, self.y_std = y_scaler

    def x_transform(self, x):
        x_mid = np.log(x)

        # Normalise and return
        return (x_mid - self.x_mean) / self.x_std

    def x_invert(self, x):
        #return np.exp(x)
        return np.exp(x * self.x_std + self.x_mean)
    
    def y_transform(self, y):
        return (y - self.y_mean) / self.y_std
    
    def y_invert(self, y):
        return y * self.y_std + self.y_mean

    @property
    def x(self) -> np.ndarray:
        return self.x_invert(self._x_data)

    @x.setter
    def x(self, value: np.ndarray):
        self._x_data = self.x_transform(value)

    @property
    def y(self) -> np.ndarray:
        return self._y_data
    
    @y.setter
    def y(self, value: np.ndarray):
        self._y_data = value

    def append_data(self, x, y, ystd):
        x    = np.array(   x).flatten()
        y    = np.array(   y).flatten()
        ystd = np.array(ystd).flatten()
        
        if not hasattr(self, '_x_data'):
            self._x_data = self.x_transform(x)
            self._y_data = self.y_transform(y)
            self._ystd_data = self.y_transform(ystd)
        else:
            self._x_data = np.vstack([self._x_data, self.x_transform(x)])
            self._y_data = np.concatenate([self._y_data, self.y_transform(y)])
            self._ystd_data = np.concatenate([self._ystd_data, self.y_transform(ystd)])

class WaitPerRankModellerMixin(Modeller):
    @property
    def cost(self) -> float:
        comm, rank, root = get_comm_rank_root()

        cost_means = np.array(self.lost_time[0], dtype=np.float32)
        cost_stds  = np.array(self.lost_time[1], dtype=np.float32)

        return cost_means, cost_stds


class ElementsPerNonwaitModellerMixin(Modeller):
    @property
    def weight(self):
        return int(self.stats["weight"][2])


class PerformanceModellerMixin(Modeller):
    @property
    def cost(self) -> float:
        comm, rank, root = get_comm_rank_root()

        return np.array(comm.allgather(self.othertime), dtype=np.float32)

class LoadingPerRankModellerMixin(Modeller):
    @property
    def parameter(self) -> np.ndarray[int]:
        comm, rank, root = get_comm_rank_root()

        return np.array(self.Nₑ, dtype=np.int32)

class LoadingsModellerMixin(LoadingPerRankModellerMixin, Modeller):
    @property
    def parameters(self) -> np.ndarray[int]:
        comm, rank, root = get_comm_rank_root()

        return np.array(comm.allgather(self.parameter), dtype=np.int32)

class NoModeller(Modeller):
    def __init__(self, intg):
        super().__init__(intg)

class GPModeller(Modeller):
    def __init__(self, intg):
        super().__init__(intg)
        self.length_scale = 1.0
        self.noise = 1e-5

    def rbf_kernel(self, x1, x2):
        """Compute the RBF kernel between x1 and x2."""
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(1, -1)
        sqdist = (x1 - x2) ** 2
        return np.exp(-0.5 * sqdist / self.length_scale ** 2)

    def rbf_kernel_derivative(self, x1, x2):
        """Compute the derivative of the RBF kernel with respect to x1."""
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(1, -1)
        K = self.rbf_kernel(x1, x2)
        dK_dx1 = (x2 - x1) / (self.length_scale ** 2) * K
        return dK_dx1

    def fit_gp(self):
        """Fit the GP model to the data."""
        X = self._x_data.flatten()
        Y = self._y_data.flatten()

        # Compute the covariance matrix
        K = self.rbf_kernel(X, X)

        # Add noise variance to the diagonal
        K += self.noise_variance * np.eye(len(X))

        # Compute the Cholesky decomposition for numerical stability
        self.L = np.linalg.cholesky(K)
        self.X_train = X
        self.Y_train = Y

        self.is_fitted = True

    def predict_gp_with_derivative(self, x_new):
        """Predict the GP mean, standard deviation, and derivative at new points."""
        if not self.is_fitted:
            raise ValueError("GP model is not fitted yet.")

        x_new = np.array(x_new).flatten()
        x_new_transformed = self.x_transform(x_new)

        X_train = self.X_train
        Y_train = self.Y_train

        # Compute cross-covariance between new points and training data
        K_s = self.rbf_kernel(x_new_transformed, X_train)

        # Compute the covariance matrix of the new points
        K_ss = self.rbf_kernel(x_new_transformed, x_new_transformed)

        # Solve for alpha using the Cholesky factorization
        L = self.L
        y = Y_train

        # Solve L * v = y
        v = np.linalg.solve(L, y)
        # Solve L.T * alpha = v
        alpha = np.linalg.solve(L.T, v)

        # Compute the posterior mean
        mu_s = K_s.dot(alpha)

        # Compute predictive variance
        # Solve L * w = K_s.T
        w = np.linalg.solve(L, K_s.T)
        # Compute the posterior covariance
        cov_s = K_ss - w.T.dot(w)
        std_s = np.sqrt(np.diag(cov_s))

        # Compute the derivative of K_s with respect to x_new_transformed
        dK_s_dx_new_transformed = self.rbf_kernel_derivative(x_new_transformed, X_train)

        # Compute the derivative of the posterior mean with respect to x_new_transformed
        dmu_s_dx_new_transformed = dK_s_dx_new_transformed.dot(alpha)

        # Adjust for the chain rule due to the x_transform (log transformation)
        dx_new_transformed_dx_new = 1 / x_new  # Derivative of log(x) is 1/x
        dmu_s_dx_new = dmu_s_dx_new_transformed * dx_new_transformed_dx_new

        # If y_transform is not identity, adjust the derivative accordingly
        # For now, y_transform is identity, so no adjustment is needed

        # Return the inverse-transformed mean, standard deviation, and derivative
        return self.y_invert(mu_s), std_s, dmu_s_dx_new

class IntegratorPerformanceLoadingGPModeller(IntegratorPerformanceObserver, 
                                             LoadingsModellerMixin,
                                             PerformanceModellerMixin,
                                             GPModeller
                                             ):
    def __init__(self, intg):
        super().__init__(intg)

class IntegratorWaitPerRankModeller(IntegratorPerformanceObserver, 
                                    LoadingPerRankModellerMixin,
                                    WaitPerRankModellerMixin,
                                    NoModeller
                                    ):
    def __init__(self, intg):
        super().__init__(intg)

class IntegratorNonWaitPerRankModeller(IntegratorPerformanceObserver, 
                                       LoadingPerRankModellerMixin,
                                       ElementsPerNonwaitModellerMixin,
                                       NoModeller
                                    ):
    def __init__(self, intg):
        super().__init__(intg)
