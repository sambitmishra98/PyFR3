import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model

import csv

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qKnowledgeGradient

from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize

from pyfr.optimisers.base import BaseGlobalOptimiser

class BoTorch(BaseGlobalOptimiser):
    name = 'botorch'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.acqf_list = self.cfg.getliteral(cfgsect, 'acqf-list')
        print(f'{self.acqf_list = }')

        bounds = self.cfg.getliteral(cfgsect, 'bounds')
        self.bounds = torch.tensor(bounds).T
        print(f'{self.bounds = }')

        self.dataset = {'parameters': [], 'costs': []}

    def __call__(self, intg):
        super().__call__(intg)

        # Currently, BO works with only one cost
        if len(self.costs) > 1:
            raise ValueError("More than one cost function specified.")

        # I have mentioned runtime as the cost function in the config file
        # I want to be more generic, since we only have one cost
        
        print(self.costs)

        with open('data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            # Each m is a list of 4 elements, so we need to store as m0-p, m0-u, m0-v, m1-p, m1-u, etc.
            csvwriter.writerow(['iter', 'tcurr', 'm0-p', 'm0-u', 'm0-v',
                                                 'm1-p', 'm1-u', 'm1-v',
                                                 'm2-p', 'm2-u', 'm2-v',
                                                 'm3-p', 'm3-u', 'm3-v',])
            
            # Write data
            for row in intg.pseudointegrator.pintg.pseudostep_multipinfo:
                new_row = row[0:2]
                for m in row[2:]:
                    new_row += m
                csvwriter.writerow(new_row)

        if self.costs['runtime'][0] is None:
            return

        #self._store_past_in_dataset()
        #self._create_and_fit_GP_model()
        
        #self.parameters = self._get_next_candidate()
        #print(f'{self.parameters = }')

        #print(f'dataset: {self.dataset}')

        #self._post_call()
        #print('NEW CYCLE-0: ', intg.pseudointegrator.cstepsf_list)

    def _plot_GP_model(self):
        # Generate test points
        x_test = torch.linspace(self.bounds[0, 0], self.bounds[1, 0], 1000).view(-1, 1)
        
        with torch.no_grad():
            posterior = self.model(x_test)
        mean = posterior.mean
        lower, upper = posterior.confidence_region()

        # Create subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))

        # Plot the GP model
        axs.scatter(self.dataset['parameters'], self.dataset['costs'], c='r', label='Data')
        axs.plot(x_test.numpy(), mean.numpy(), 'k:', label='Prediction (mean)')
        axs.fill_between(x_test.numpy().ravel(), lower.numpy(), upper.numpy(), alpha=0.2)
        axs.set_title('Gaussian Process Model')
        axs.set_xlabel('Parameter')
        axs.set_ylabel('Cost')
        axs.legend()

        plt.tight_layout()
        plt.savefig('gp_and_acq_model.png')

    def _create_and_fit_GP_model(self):
        train_X = torch.tensor(self.dataset['parameters'], 
                               dtype=torch.float64
                               ).view(-1, len(self.parameter_names))
        train_Y = torch.tensor(self.dataset['costs'], 
                               dtype=torch.float64
                               ).view(-1, 1)
        
        self.model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(train_X)


    def _get_next_candidate(self):
        # Choose the acquisition function based on acqf_list
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

        # Define the acquisition function
        if acq_func == 'ei':
            acq_function = ExpectedImprovement(self.model, 
                                               torch.min(torch.tensor(self.dataset['costs'])))
        elif acq_func == 'kg':
            acq_function = qKnowledgeGradient(self.model, 
                                              num_fantasies=1,
                                              )
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        # Optimize the acquisition function
        print(f'{self.bounds = }')
        bounds_list = [(lb.item(), 
                        ub.item()) for lb, ub in zip(self.bounds[0, :], 
                                                     self.bounds[1, :])]

        res = minimize(lambda x: -acq_function(torch.tensor([x])).item(), 
               x0=[1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.7], bounds=bounds_list)

        return res.x.tolist()
    
    def _store_past_in_dataset(self):
        prev_cost = self.costs['runtime'][0]
        prev_candidate = [self.parameters[pn] for pn in self.parameter_names]
        
        
        self.dataset['parameters'].append(prev_candidate)
        self.dataset['costs'].append(prev_cost)
