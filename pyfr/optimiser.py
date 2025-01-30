import numpy as np
from scipy.optimize import minimize

from scipy.stats import norm

from pyfr.modeller import (IntegratorPerformanceLoadingGPModeller, 
                           IntegratorWaitPerRankModeller,
                           IntegratorNonWaitPerRankModeller)


class PerformanceMaximiser(IntegratorPerformanceLoadingGPModeller):
    
    def __init__(self, intg):
        super().__init__(intg)
    
    def next_candidate(self):

        def ei(x):
            mean, std = self.gp.predict(x.reshape(1,-1), return_std=True)
            best_y = np.max(self.y)
            z = (mean-best_y) / (std + 1e-9)
            ei =(mean-best_y) * norm.cdf(z) + std * norm.pdf(z)
            return -ei  # Minimizing the negative Expected Improvement

        result = minimize(
            ei,
            x0=np.random.uniform(self.bound_lower, self.bound_upper),
            bounds=list(zip(self.bound_lower, self.bound_upper)),
            method='L-BFGS-B'
        )
        return result.x
    
    def best_candidate_prediction(self):
        # Choose the best as per maximum of posterior mean
        def mpm(x):
            return -self.gp.predict(x.reshape(1,-1), return_std=False)
        
        result = minimize(
            mpm,
            x0=np.random.uniform(self.bound_lower, self.bound_upper),
            bounds=list(zip(self.bound_lower, self.bound_upper)),
            method='L-BFGS-B'
        )
        return result.x

class WaitMinimiser(IntegratorWaitPerRankModeller):
    
    def __init__(self, intg):
        super().__init__(intg)
    

class WaitMinimiser(IntegratorNonWaitPerRankModeller):
    
    def __init__(self, intg):
        super().__init__(intg)
