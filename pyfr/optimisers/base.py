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
        self.get_parameters = {}
        self.set_parameters = {}
        self.initialise_parameters_calls(intg)

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
        print(self.costlists)

    def initialise_parameters_calls(self, intg):

        for parameter_name in self.parameter_names:
            if parameter_name.startswith('pmultigrid-'):
                index = int(parameter_name.split('-')[1])
                
                @staticmethod
                def get_parameter(intg=intg, index=index):
                    return intg.pseudointegrator.cstepsf_list[0][index]

                @staticmethod
                def set_parameter(y, intg=intg, index=index):
                    for i in range(len(intg.pseudointegrator.cstepsf_list)):
                        intg.pseudointegrator.cstepsf_list[i][index] = y

                self.get_parameters[parameter_name] = get_parameter
                self.set_parameters[parameter_name] = set_parameter
                
            elif parameter_name == 'ac-zeta':
                @staticmethod
                def get_parameter(intg=intg):
                    return intg.pseudointegrator.pintg.system.ac_zeta

                @staticmethod
                def set_parameter(y, intg=intg):
                    intg.pseudointegrator.pintg.system.ac_zeta = y

                self.get_parameters[parameter_name] = get_parameter
                self.set_parameters[parameter_name] = set_parameter
                
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
