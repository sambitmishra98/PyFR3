# logger_utils.py

import logging
import functools
import time
from collections.abc import Mapping, Sequence

import numpy as np
from termcolor import colored

from pyfr.mpiutil import get_comm_rank_root


def initialise_logger(name: str, level: int) -> logging.Logger:
    """
    Initialize a logger with the specified name and logging level.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    comm, rank, root = get_comm_rank_root()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Ensure the logging directory exists
    import os
    os.makedirs('logging', exist_ok=True)

    file_handler = logging.FileHandler(f'logging/{name}-{rank}.log')
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("_______________________________________________")
    logger.info(f"{name} initialized.")
    return logger


def log_method_times(method):
    """
    Decorator to log the execution time of methods.

    Args:
        method (callable): The method to decorate.

    Returns:
        callable: Wrapped method with logging.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        tstart = time.perf_counter()
        result = method(self, *args, **kwargs)
        tdiff = time.perf_counter() - tstart

        logger = self._logger
        mname = method.__name__

        if tdiff > 10:
            logger.critical(f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        elif tdiff > 5:
            logger.error(f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        elif tdiff > 1:
            logger.warning(f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        else:
            logger.debug(f"walltime: \t {mname}: {tdiff:.4f} s")
        return result
    return wrapper


def cprint(text, name: str = 'blank') -> None:
    """
    Synchronized colored print across MPI ranks.

    Args:
        text (Any): The content to print.
        name (str, optional): Label for the printed content. Defaults to 'blank'.
    """
    comm, rank, root = get_comm_rank_root()
    rank_colors = ['green', 'blue', 'red', 'yellow', 'cyan', 'magenta', 'white']
    color = rank_colors[rank % len(rank_colors)]
    rank_color = lambda msg: colored(msg, color=color)

    def format_recursive(obj, indent: int = 0) -> str:
        """
        Recursively format complex objects for pretty printing.

        Args:
            obj (Any): The object to format.
            indent (int, optional): Current indentation level. Defaults to 0.

        Returns:
            str: Formatted string representation.
        """
        ind = '    ' * indent  # 4 spaces per indentation level
        if isinstance(obj, Mapping):
            lines = []
            for k, v in obj.items():
                formatted_v = format_recursive(v, indent + 1)
                lines.append(f"{ind}{k}: {formatted_v}")
            return '{\n' + '\n'.join(lines) + f'\n{ind}}}'
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            lines = []
            for v in obj:
                formatted_v = format_recursive(v, indent + 1)
                lines.append(f"{ind}{formatted_v}")
            return '[\n' + '\n'.join(lines) + f'\n{ind}]'
        elif isinstance(obj, np.ndarray):
            with np.printoptions(
                precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
            ):
                array_str = np.array2string(obj, separator=', ')
            return array_str
        else:
            return repr(obj)

    # Ensure outputs are printed in order by rank without sleep
    for r in range(comm.size):
        if rank == r:
            print(rank_color(f"{name}"))

            if isinstance(text, dict):
                formatted_text = format_recursive(text)
                print(rank_color(formatted_text))

            elif isinstance(text, list) and all(isinstance(t, np.ndarray) for t in text):
                with np.printoptions(
                    precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
                ):
                    for t in text:
                        print(rank_color(t), end="\n\n")

            elif isinstance(text, np.ndarray):
                with np.printoptions(
                    precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
                ):
                    print(rank_color(text))

            else:
                # Use recursive formatter for complex objects
                formatted_text = format_recursive(text)
                print(rank_color(formatted_text))

            if rank == comm.size - 1:
                print("-" * 100)
        # Synchronize ranks with a slight delay to maintain order
        time.sleep(r / 100)
