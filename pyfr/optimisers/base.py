import numpy as np

def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if outf.tell() == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


class BaseOptimiser:
    
    name = None
    systems = None
    formulations = None
    
    def __init__(self, intg, cfgsect, suffix = None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        cost_names = self.cfg.get(cfgsect, 'cost').split()
        self.costlists = {cost_name: np.zeros((1, 1)) for cost_name in cost_names}
        self.process_costs = {}
        self.prev_costs = {cost_name: np.zeros((1, 1)) for cost_name in cost_names}
        self.initialise_costs(cost_names)

        self.parameter_names = self.cfg.get(cfgsect, 'parameter').split()

        # Store the entire dataset of all parameters.
        self._initialise_parameters(intg)

        #self.get_parameters = {}
        #self.set_parameters = {}
        #self.initialise_parameters_calls(intg)

    def _initialise_parameters(self, intg):

        intg.parameters = {}
        self.parameters_dataset = {}

        default_vary_with_levels            = True
        default_vary_with_stages            = True
        default_vary_with_pseudo_iterations = True

        for parameter_name in self.parameter_names:
            if parameter_name.startswith('psmoothing-'):
                index = int(parameter_name.split('-')[1])

                if default_vary_with_levels:
                    levels = self.cfg.getint('solver', 'order') + 1
                else:
                    levels = 1

                if default_vary_with_stages:
                    stages = 3
                else:
                    stages = 1

                if default_vary_with_pseudo_iterations:
                    pseudo_iterations = self.cfg.getint('solver-time-integrator', 'pseudo-niters-max')
                else:
                    pseudo_iterations = 1

                intg.parameters[parameter_name] = 2.5 * np.ones((stages, 
                                                            levels,
                                                            pseudo_iterations, 
                                                            ))

                # Let us instead have the numbers from 1 to largest, so that we can monitor indices
                intg.parameters[parameter_name] = np.arange(1.0, 1.0 + stages*levels*pseudo_iterations).reshape((stages, 
                                                            levels,
                                                            pseudo_iterations, 
                                                            ))
                
        self.parameters_dataset[parameter_name] = [intg.parameters]

    def __call__(self, intg):
        self.update_costlists(intg)

    def _post_call(self):
        self.prev_costs = self.costs
        self.costlists = {cost_name: np.zeros((1, 1)) for cost_name in self.costlists}

    @property
    def costs(self):
        costs = {}
        for cost_name in self.costlists:
            costs[cost_name] = self.process_costs[cost_name](self)
        return costs

    # If only one cost, then we can just use the following getter
    @property
    def cost(self):
        return self.costs[list(self.costlists.keys())[0]]

    @property
    def prev_cost(self):
        return self.prev_costs[list(self.costlists.keys())[0]]
    
    def initialise_costs(self, cost_names):

        for cost_name in cost_names:

            if cost_name == 'runtime':

                def process_cost(self, n_skip = 1, n_capture = 4):
                    if len(self.costlists[cost_name]) < n_skip + n_capture:
                        return None, None
                    else:
                        captured = self.costlists[cost_name][n_skip:]
                        cost = np.mean(captured)
                        cost_err  = np.std(captured)/np.sqrt(len(captured))/cost

                        return cost, cost_err

                self.process_costs[cost_name] = process_cost

            elif cost_name == 'integral-absolute-divergence':

                def process_cost(self, n_skip = 1, n_capture = 4):
                    if len(self.costlists[cost_name]) < n_skip + n_capture:
                        return None, None
                    else:
                        captured = self.costlists[cost_name][n_skip:]
                        cost = np.mean(captured)
                        cost_err  = np.std(captured)/np.sqrt(len(captured))/cost

                        return cost, cost_err

                self.process_costs[cost_name] = process_cost

    def update_costlists(self, intg):
        for cost_name in self.costlists:
            if cost_name == 'runtime':
                self.costlists[cost_name] = np.append(self.costlists[cost_name],
                                                      intg.performanceinfo)
            elif cost_name == 'integral-absolute-divergence':
                self.costlists[cost_name] = np.append(self.costlists[cost_name],
                                                      intg.integral)
            elif cost_name == 'modalresidualnorms':
                # Get modal residuals from an error register 
                # as an [order+1 × order+1] matrix
                # And plot all of them
                # We need [level, csteps, (1/r)×(∂r/(i∂τ))]

                self.costlists[cost_name] = np.append(self.costlists[cost_name],
                                                      intg.resnorms)

            else:
                raise ValueError(f'Cost {cost_name} is not recognised')
        print(self.costlists)

    @property
    def parameters(self):
        parameters = {}
        for parameter_name in self.parameter_names:
            parameters[parameter_name] = self.get_parameters[parameter_name]()
        return parameters
    
    @parameters.setter
    def parameters(self, ys):
        print('we are setting the parameters')
        print(f'parameter: {self.parameter_names} --> {ys}')
        for parameter_name, y in zip(self.parameter_names,ys):

            print(parameter_name, y)
            self.set_parameters[parameter_name](y)

    # If only one parameter, then we can just use the following setter and getter
    @property
    def parameter(self):
        return self.parameters[self.parameter_names[0]]
    
    @parameter.setter
    def parameter(self, y):
        self.parameters = [y,]


class BaseGlobalOptimiser(BaseOptimiser):
    prefix = 'global'
    
    
class BaseLocalOptimiser(BaseOptimiser):
    prefix = 'local'
