import matplotlib.pyplot as plt
import numpy as np

from pyfr.observer import IntegratorPerformanceObserver
from pyfr.mpiutil import get_comm_rank_root

class Modeller:
    def __init__(self, intg):
        self.intg = intg

    def append_data(self, x, y, ystd):
        x    = np.array(x).flatten()
        y    = np.array(y).flatten()
        ystd = np.array(ystd).flatten()
        
        if not hasattr(self, '_x_data'):
            self._x_data = x
            self._y_data = y
            self._ystd_data = ystd
        else:
            self._x_data    = np.vstack([self._x_data, x])
            self._y_data    = np.concatenate([self._y_data, y])
            self._ystd_data = np.concatenate([self._ystd_data, ystd])

class PerformanceModellerMixin(Modeller):
    @property
    def costs(self) -> float:
        comm, rank, root = get_comm_rank_root()

        return np.array(comm.allgather(self.othertime), dtype=np.float32)

class LoadingModellerMixin(Modeller):
    @property
    def parameters(self) -> list[int]:
        comm, rank, root = get_comm_rank_root()

        return np.array(comm.allgather(self.Nₑ), dtype=np.int32)

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
        x_new_transformed = x_new

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
        dK_s_dx_new_transformed = self.rbf_kernel_derivative(x_new_transformed, X_train)
        return self.y_invert(mu_s), std_s, dK_s_dx_new_transformed.dot(alpha)

class IntegratorPerformanceLoadingGPModeller(IntegratorPerformanceObserver, 
                                             LoadingModellerMixin,
                                             PerformanceModellerMixin,
                                             GPModeller
                                             ):
    def __init__(self, intg):
        super().__init__(intg)
