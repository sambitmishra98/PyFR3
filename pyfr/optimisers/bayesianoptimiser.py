import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

import matplotlib.pyplot as plt

from pyfr.optimisers.base import BaseGlobalOptimiser

class BayesianOptimiser(BaseGlobalOptimiser):
    name = 'bayesopt'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.acqf_list = self.cfg.getliteral(cfgsect, 'acqf-list')
        print(f'{self.acqf_list = }')

        bounds = self.cfg.getliteral(cfgsect, 'bounds')
        self.bounds = np.array(bounds).reshape(2, -1)

        self.dataset = {'parameters': [], 'costs': []}

    def __call__(self, intg):
        super().__call__(intg)

        if self.costs['runtime'][0] is None:
            return

        self._store_past_in_dataset()
        self._create_and_fit_GP_model()
        self._plot_GP_model()
        next_candidate = self._get_next_candidate()

        self.parameters = [next_candidate,]
        print(f'dataset: {self.dataset}')

        self._post_call()

    def _plot_GP_model(self):
        # Generate test points
        x_test = np.linspace(self.bounds[0, 0], self.bounds[1, 0], 1000).reshape(-1, 1)
        x_test_scaled = self.scaler_X.transform(x_test)

        # Get model predictions
        mean, std = self.model.predict(x_test_scaled, return_std=True)

        # Unscale the predictions
        mean = mean.reshape(-1, 1)
        mean = self.scaler_Y.inverse_transform(mean)
        std = std * self.scaler_Y.scale_

        # Using Expected Improvement (EI) acquisition function
        def ei(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            mean, std = self.model.predict(x, return_std=True)
            best_y = np.min(self.dataset['costs'])
            z = (best_y - mean) / std
            return -(std * (z * norm.cdf(z) + norm.pdf(z)))
        
        acq_values = -ei(x_test)

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the GP model
        axs[0].scatter(self.dataset['parameters'], self.dataset['costs'], c='r', label='Data')
        axs[0].plot(x_test, mean, 'k:', label='Prediction (mean)')
        axs[0].fill_between(x_test.ravel(), mean.ravel() - 1.96 * std, mean.ravel() + 1.96 * std, alpha=0.2)
        axs[0].set_title('Gaussian Process Model')
        axs[0].set_xlabel('Parameter')
        axs[0].set_ylabel('Cost')
        axs[0].legend()

        # Plot the acquisition function
        axs[1].plot(x_test, acq_values, 'g--', label='Acquisition Function (EI)')
        axs[1].set_title('Acquisition Function')
        axs[1].set_xlabel('Parameter')
        axs[1].set_ylabel('Acquisition Value')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig('gp_and_acq_model.png')

    def _create_and_fit_GP_model(self):
        kernel = C(1.0) * Matern(length_scale=0.1, nu = 2.5)  # Increase upper bound
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1000)  # Increase number of restarts
        
        # Standardize the data
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        train_X = np.array(self.dataset['parameters']).reshape(-1, 1)
        train_Y = np.array(self.dataset['costs']).reshape(-1, 1)
        
        train_X = self.scaler_X.fit_transform(train_X)
        train_Y = self.scaler_Y.fit_transform(train_Y)
        
        self.model.fit(train_X, train_Y)

    def _get_next_candidate(self):

        # Using Upper Confidence Bound (UCB) acquisition function
        def ucb(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            mean, std = self.model.predict(x, return_std=True)
            return -(mean + 0.1 * std)

        # Using Expected Improvement (EI) acquisition function
        def ei(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            mean, std = self.model.predict(x, return_std=True)
            best_y = np.min(self.dataset['costs'])
            z = (best_y - mean) / std
            return -(std * (z * norm.cdf(z) + norm.pdf(z)))

        # Using Probability of Improvement (PI) acquisition function
        def pi(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            mean, std = self.model.predict(x, return_std=True)
            best_y = np.min(self.dataset['costs'])
            z = (best_y - mean) / std
            return -norm.cdf(z)

        # Using Maximum Variance (MV) acquisition function
        def mv(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            _, std = self.model.predict(x, return_std=True)
            return -std

        # Using Mean-Std (MS) acquisition function
        def ms(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            mean, std = self.model.predict(x, return_std=True)
            return -(mean + std)

        # Using Random Sampling as acquisition function
        def random_sampling(x):
            return -np.random.rand()

        # Using Thompson Sampling as acquisition function
        def thompson_sampling(x):
            x = np.array(x).reshape(-1, 1)
            x = self.scaler_X.transform(x)
            samples = self.model.sample_y(x, 1)
            return -samples[0, 0]

        num_tested = len(self.dataset['parameters'])
        acq_func = None

        total = 0
        for func_name, count in self.acqf_list:
            total += count
            if num_tested < total:
                acq_func = func_name
                break

        if acq_func is None:
            raise ValueError("No acquisition function specified for the current number of tested candidates.")

        acq_funcs = {
            'ucb': ucb,
            'ei': ei,
            'pi': pi,
            'mv': mv,
            'ms': ms,
            'random': random_sampling,
            'thompson': thompson_sampling
        }

        if acq_func not in acq_funcs:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        chosen_acq_func = acq_funcs[acq_func]

        # If random sampling or thompson sampling, we don't need to optimize.
        if acq_func in ['random', 'thompson']:
            return chosen_acq_func(None)


        res = minimize(chosen_acq_func, x0=[0.5], 
                                    bounds=self.bounds.T, 
                                    options={'maxiter': 200})

        print(f"We used {acq_func} acquisition function. and got {res.x[0]} as the next candidate.")

        return res.x[0]

    def _store_past_in_dataset(self):
        prev_cost = self.costs['runtime'][0]
        prev_candidate = self.parameters['pmultigrid']
        
        self.dataset['parameters'].append(prev_candidate)
        self.dataset['costs'].append(prev_cost)
