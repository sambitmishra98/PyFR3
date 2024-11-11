from pyfr.plugins.base import BaseSolnPlugin
from pyfr.mpiutil import get_comm_rank_root, mpi
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import ast
import logging

class InspectPlugin(BaseSolnPlugin):
    name = 'inspect'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # MPI setup
        comm, rank, root = get_comm_rank_root()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        
        # Prevent this logger from propagating messages to the root logger
        self._logger.propagate = False
        
        # Create a file handler
        log_file_path = logging.FileHandler('logging/plugin-inspect.log')
        log_file_path.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file_path.setFormatter(formatter)
        
        # Add the handler to the logger
        self._logger.addHandler(log_file_path)
        
        self._logger.info("Inspect initialized.") 

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

        # Get output file name template
        self.outfile = self.cfg.get(cfgsect, 'file')
        base, ext = os.path.splitext(self.outfile)
        self.outfile = f"{base}_{suffix}{ext}" if suffix else self.outfile

        # Get attributes to monitor
        self.attributes = self.cfg.getliteral(cfgsect, 'attributes')

        # Get plot-rankwise operations
        self.plot_rankwise = self.cfg.getliteral(cfgsect, 'plot-rankwise', None)
        if self.plot_rankwise:
            if not isinstance(self.plot_rankwise, list):
                self._logger.warning("'plot-rankwise' must be a list. Ignoring.")
                self.plot_rankwise = None
            elif len(self.plot_rankwise) != len(self.attributes):
                self._logger.warning("len('plot-rankwise') != len('attributes'). Ignoring.")
                self.plot_rankwise = None
            else:
                # Check for valid operations
                valid_ops = {'sum', 'mean', 'min', 'max', 'each'}
                for op in self.plot_rankwise:
                    if op not in valid_ops:
                        self._logger.warning(f"Invalid operation '{op}' in 'plot-rankwise'. Ignoring.")
                        self.plot_rankwise = None
                        break

        # X and Y labels, title, grids, legend labels, caption, if-log
        self.title     = self.cfg.get(cfgsect, 'title', None)
        self.subtitle  = self.cfg.get(cfgsect, 'subtitle', None)
        self.xlabel    = self.cfg.get(cfgsect, 'xlabel', None)
        self.ylabel    = self.cfg.get(cfgsect, 'ylabel', None)
        self.add_grids = self.cfg.getbool(cfgsect, 'add-grids', False)
        self.legend    = self.cfg.getliteral(cfgsect, 'legend', None)
        self.caption   = self.cfg.get(cfgsect, 'caption', '')
        self.iflog     = self.cfg.getbool(cfgsect, 'if-log', False)

        # Choose x-axis: 'step' or 'time' (default to 'step')
        self.xaxis = self.cfg.get(cfgsect, 'xaxis', 'step')

        if self.legend:
            if isinstance(self.legend, dict):
                # Map attribute names to labels
                self.legend = [self.legend.get(attr, attr) for attr in self.attributes]
            elif isinstance(self.legend, list):
                if len(self.legend) != len(self.attributes):
                    self._logger.warning("Length of 'legend' does not match 'attributes'. Ignoring legend labels.")
                    self.legend = None
            else:
                self._logger.warning("'legend' must be a list or a dictionary. Ignoring legend labels.")
                self.legend = None
        else:
            self.legend = self.attributes  # Default legend is attribute names

        # Initialize data storage
        self.data = [[] for _ in self.attributes]
        self.steps = []
        self.times = []

        # Store MPI communicator and rank info
        self.comm = comm
        self.rank = rank
        self.root = root

    def resolve_attr(self, obj, attr_path):
        """
        Resolve attribute path on an object
        - supporting nested attributes and dictionary keys.
        - The attr_path can include:
            - Attributes: separated by dots, e.g., 'system.ndims'
            - Dictionary access: using [key], e.g., '_plugin_wtimes[("writer", None)]'

        Example:
            resolve_attr(intg, '_plugin_wtimes[("writer", None)]')

        Note:
            - Dictionary keys should be strings, integers, or tuples of strings/integers.
            - For safety, only allow literal evaluation of dictionary keys.
        """
        # Regular expression to match attribute or dictionary access
        regex = r'''
            (?P<attr>\w+)             # Match attribute name
            (?:                       # Start non-capturing group
                \[                    # Match opening bracket
                (?P<key>.*?)          # Match anything (non-greedy) as key
                \]                    # Match closing bracket
            )?                        # Group is optional
            (?:\.|$)                  # Match optional dot between attributes
        '''
        tokens = re.finditer(regex, attr_path, re.VERBOSE)

        for match in tokens:
            attr = match.group('attr')
            key = match.group('key')

            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                obj = obj[attr]
            else:
                self._logger.warning(
                    f"'{attr}' missing in object of type '{type(obj)}'.")
                return np.nan

            if key is not None:
                # Evaluate the key safely
                try:
                    key = ast.literal_eval(key)
                except Exception:
                    self._logger.warning(f"Invalid dictionary key: {key}")
                    return np.nan

                try:
                    obj = obj[key]
                except (KeyError, TypeError):
                    self._logger.warning(f"Key '{key}' missing from dict.")
                    return np.nan

        return obj

    def __call__(self, intg):

        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        # Record the current step and time
        self.steps.append(intg.nacptsteps)
        self.times.append(intg.tcurr)

        # Collect data for each attribute
        local_values = []
        for idx, attr in enumerate(self.attributes):
            # Get attribute value
            value = self.resolve_attr(intg, attr)

            # If it's a numpy array, process it (e.g., take mean)
            if isinstance(value, np.ndarray):
                value = value.mean()
            elif not isinstance(value, (int, float)):
                self._logger.warning(f"Attribute '{attr}' is not a numeric type.")
                value = np.nan

            local_values.append(value)

        # Perform MPI reduction if plot_rankwise is specified
        if self.plot_rankwise:
            for idx, op in enumerate(self.plot_rankwise):
                value = local_values[idx]
                if op == 'sum':
                    global_value = self.comm.allreduce(value, op=mpi.SUM)
                    if self.rank == self.root:
                        self.data[idx].append(global_value)
                elif op == 'mean':
                    total = self.comm.allreduce(value, op=mpi.SUM)
                    size = self.comm.allreduce(1, op=mpi.SUM)
                    global_value = total / size
                    if self.rank == self.root:
                        self.data[idx].append(global_value)
                elif op == 'min':
                    global_value = self.comm.allreduce(value, op=mpi.MIN)
                    if self.rank == self.root:
                        self.data[idx].append(global_value)
                elif op == 'max':
                    global_value = self.comm.allreduce(value, op=mpi.MAX)
                    if self.rank == self.root:
                        self.data[idx].append(global_value)
                elif op == 'each':
                    # Gather data from all ranks to root
                    all_values = self.comm.gather(value, root=self.root)
                    if self.rank == self.root:
                        self.data[idx].append(all_values)
                else:
                    self._logger.warning(f"Unsupported operation '{op}' in 'plot-rankwise'.")
                    if self.rank == self.root:
                        self.data[idx].append(np.nan)
        else:
            # No plot_rankwise specified, collect local values
            if self.rank == self.root:
                for idx, value in enumerate(local_values):
                    self.data[idx].append(value)

        # Output data
        if self.rank == self.root:
            self._output_data(intg)
            
        # Update the last output time
        self.tout_last = intg.tcurr

    def _output_data(self, intg):
        # Only root rank outputs data
        if self.rank != self.root:
            return

        # Determine output file name with suffix
        base, ext = os.path.splitext(self.outfile)

        if ext == '.csv':
            self._write_csv(self.outfile)
        elif ext == '.png':
            # Prepare x-axis data
            xdata = self.steps if self.xaxis == 'step' else self.times
            xlabel = self.xlabel or ('Step' if self.xaxis == 'step' else 'Time')
            self._plot_png(self.outfile, xdata, xlabel)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _write_csv(self, outfile):
        # Output data to CSV
        data_dict = {}
        data_dict['step'] = self.steps
        data_dict['time'] = self.times

        for idx, attr in enumerate(self.attributes):
            ydata = self.data[idx]
            op = self.plot_rankwise[idx] if self.plot_rankwise else None
            label = self.legend[idx] if self.legend else attr

            if op == 'each':
                # ydata is a list of lists: one per time step, each containing values from all ranks
                # Transpose ydata to get list per rank
                ydata_per_rank = list(map(list, zip(*ydata)))  # Now ydata_per_rank[rank][time]
                num_ranks = len(ydata_per_rank)
                for rank_idx in range(num_ranks):
                    col_name = f"{label}_rank{rank_idx}"
                    data_dict[col_name] = ydata_per_rank[rank_idx]
            else:
                # ydata is a list of values
                col_name = f"{label} ({op})" if op else label
                data_dict[col_name] = ydata

        df = pd.DataFrame(data_dict)
        df.to_csv(outfile, index=False)
        self._logger.info(f"Data written to CSV file '{outfile}'.")

    def _plot_png(self, outfile, xdata, xlabel):
        # Plot data and save as PNG

        # Set up matplotlib to use default fonts
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plt.rcParams['text.parse_math'] = False  # Disable mathtext parsing

        # Set figure size and dpi for high-quality output
        plt.figure(figsize=(10, 4), dpi=300)

        # Plot each attribute
        for idx, attr in enumerate(self.attributes):
            ydata = self.data[idx]
            op = self.plot_rankwise[idx] if self.plot_rankwise else None
            label = self.legend[idx] if self.legend else attr

            # Replace underscores with spaces to avoid issues
            label = label.replace('_', ' ')

            # Add operation to label if applicable
            if op and op != 'each':
                label = f"{label} ({op})"

            if op == 'each':
                # ydata is a list of lists: one per time step, each containing values from all ranks
                # Transpose ydata to get list per rank
                ydata_per_rank = list(map(list, zip(*ydata)))  # Now ydata_per_rank[rank][time]
                num_ranks = len(ydata_per_rank)
                for rank_idx in range(num_ranks):
                    ydata_rank = ydata_per_rank[rank_idx]
                    rank_label = f"{label} - Rank {rank_idx}"
                    plt.plot(xdata, ydata_rank, label=rank_label, 
                             marker='o', markersize=2)
            else:
                # ydata is a list of values
                plt.plot(xdata, ydata, label=label, linestyle= '--')

        # Set x and y labels
        plt.xlabel(self.xlabel or xlabel)
        plt.ylabel(self.ylabel or 'Value')

        # Set title
        if self.title:
            plt.suptitle(self.title)

        # Set subtitle
        if self.subtitle:
            plt.title(self.subtitle)

        # Add grids
        if self.add_grids:
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            plt.minorticks_on()
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0),)

        # Use log scale if specified
        if self.iflog:
            plt.yscale('log')

        # Add legend, center left outside of the plot
        plt.legend( loc='center left', bbox_to_anchor=(1.01, 0.5), 
                   borderaxespad=0.)

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # Adjust layout to make room for figure caption
        plt.tight_layout()

        # Add figure caption as text below the plot
        if self.caption:
            plt.figtext(0.5, -0.1, self.caption,
                        wrap=True,
                        horizontalalignment='center', fontsize=10)

        # Save the plot
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        self._logger.info(f"Plot saved as PNG file '{outfile}'.")
