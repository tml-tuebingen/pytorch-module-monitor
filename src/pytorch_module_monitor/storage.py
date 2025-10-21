import torch
import numpy as np
import sys
from typing import Any, Dict, List, Union, Callable
from numbers import Number


class StorageManager:
    """Manages the storage and manipulation of metrics in a dictionary.
    
    The log_dict has the structure:
    {
        step1: {"key1": value1, "key2": value2},
        step2: {"key1": value1, "key2": value2},
    }
    """
    
    def __init__(self, logger=None):
        self.log_dict = {}
        self.aggregation_fns = {}
        self.logger = logger
        
    def create_step_entry(self, step: int):
        """Create a new entry for a step in the log dict."""
        self.log_dict[step] = {}
        
    def has_step(self, step: int) -> bool:
        """Check if a step exists in the log dict."""
        return step in self.log_dict
    
    def get_step_metrics(self, step: int) -> dict:
        """Return the log dict of a specific step."""
        return self.log_dict.get(step, {})
    
    def get_all_metrics(self) -> dict:
        """Return the full log dict with all steps."""
        return self.log_dict
    
    def load_metrics(self, log_dict: dict):
        """Load a log dict."""
        if len(self.log_dict) > 0:
            raise ValueError("The current log dict is not empty. Please clear it first.")
        self.log_dict = log_dict
        
    def clear(self):
        """Clear the log dict."""
        self.log_dict = {}
        
    def log_scalar(self, step: int, key: str, value: Union[Number, torch.Tensor], aggregation_fn: Callable | None = None):
        """Log a scalar value.
        
        If the same key is logged multiple times in a step, values are stored in a list.
        """
        # Ensure step exists
        if step not in self.log_dict:
            self.log_dict[step] = {}
            
        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            value = value.item()
            
        # Verify that value is a number
        if not isinstance(value, Number):
            raise ValueError(f"log_scalar: The provided value is not a scalar. Key: {key}, Value: {value}")
        
        # Handle multiple values for the same key
        if key in self.log_dict[step]:
            if not isinstance(self.log_dict[step][key], list):
                self.log_dict[step][key] = [self.log_dict[step][key]]
            self.log_dict[step][key].append(value)
        else:
            self.log_dict[step][key] = value

        # Store aggregation function if provided
        if aggregation_fn is not None:
            self.aggregation_fns[key] = aggregation_fn
            
    def log_tensor(self, step: int, key: str, tensor: torch.Tensor, aggregation_fn: Callable | None = None):
        """Log a tensor.
        
        Tensors are stored in a list an aggregated with the provided aggregation_fn.
        """
        if step not in self.log_dict:
            self.log_dict[step] = {}
            
        # Detach and flatten the tensor
        tensor = tensor.detach().flatten()
        
        # Store in list
        if key in self.log_dict[step]:
            self.log_dict[step][key].append(tensor)
        else:
            self.log_dict[step][key] = [tensor]

        # Store aggregation function if provided
        if aggregation_fn is not None:
            self.aggregation_fns[key] = aggregation_fn


    def aggregate_step(self, step: int):
        """Aggregate values logged across micro-batches for a step.

        For lists of scalars/tensors, applies aggregation function (default: mean).
        Custom aggregation functions can return a dict for multiple aggregated values.

        Args:
            step: The step number to aggregate.
        """
        if step not in self.log_dict:
            return
            
        for k, v in list(self.log_dict[step].items()):
            if type(v) == list:
                if isinstance(v[0], torch.Tensor):
                    # List of tensors
                    values = torch.cat(v)
                    #agg_fn = lambda x: {'[mean]': torch.mean(x).item(), '[std]': torch.std(x).item()}
                    agg_fn = lambda x: torch.mean(x).item()
                       
                else:
                    # List of scalars
                    values = v
                    #agg_fn = lambda x: {'[mean]': np.mean(x), '[std]': np.std(x)}
                    agg_fn = lambda x: np.mean(x)

                # look for user-defined aggregation function
                if k in self.aggregation_fns:
                    agg_fn = self.aggregation_fns[k]

                # call the aggregation function and store results
                aggregated_values = agg_fn(values)

                if isinstance(aggregated_values, dict):
                    # dict with multiple values
                    for sub_k, sub_v in aggregated_values.items():
                        self.log_dict[step][k + sub_k] = sub_v

                    if not '' in aggregated_values: # remove original key if not explicitly included
                        del self.log_dict[step][k]
                else:
                    # a single value
                    self.log_dict[step][k] = aggregated_values
                    
        # Log info about the step
        if self.logger:
            size_kb = sys.getsizeof(self.log_dict) / 1024
            self.logger.info(
                f"Logged {len(self.log_dict[step])} keys at step {step}. "
                f"Total size of log data: {size_kb:.2f} KB"
            )
        
    def condensed_log_dict(self) -> dict:
        """Transform the log_dict structure.
        
        From:
        {
            step1: {"key1": value1, "key2": value2},
            step2: {"key1": value1, "key2": value2},
        }
        
        To:
        {
            "key1": {step1: value1, step2: value2},
            "key2": {step1: value1, step2: value2},
        }
        """
        new_dict = {}
        for step, metrics in self.log_dict.items():
            for name, val in metrics.items():
                if name not in new_dict:
                    new_dict[name] = {}
                new_dict[name][step] = val
        return new_dict
        
    def save_hdf5(self, filename: str, condensed: bool = True):
        """Save the log dict as HDF5.

        Args:
            filename: Path to save the HDF5 file.
            condensed: If True, saves in condensed format (metrics grouped by name).
        """
        import h5py
        
        log_dict = self.log_dict
        if condensed:
            log_dict = self.condensed_log_dict()
            
        with h5py.File(filename, 'w') as f:
            for parameter, value_dict in log_dict.items():
                try:
                    # Create a group for each parameter
                    group = f.create_group(parameter)
                    
                    # Save keys and values separately
                    keys = list(value_dict.keys())
                    values = list(value_dict.values())
                    
                    keys = np.array(keys, dtype=np.float64)
                    values = np.array(values, dtype=np.float64)
                    
                    group.create_dataset('keys', data=keys)
                    group.create_dataset('values', data=values)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Could not save parameter %s with keys %s and values %s: %s",
                            parameter, keys, values, e
                        )
                        
    @staticmethod
    def read_hdf5_entry_keys(filename: str) -> List[str]:
        """Read the names of the entries in an HDF5 file."""
        import h5py
        with h5py.File(filename, 'r') as f:
            return list(f.keys())
            
    @staticmethod
    def read_hdf5_entry(filename: str, entry_key: str):
        """Read a single entry from HDF5 file by its outer key."""
        import h5py
        
        with h5py.File(filename, 'r') as f:
            key_str = str(entry_key)
            
            if key_str not in f:
                return None
                
            inner_dict = {}
            for key in f[key_str]:
                value = f[key_str][key][()]
                if isinstance(value, np.generic):
                    value = value.item()
                inner_dict[key] = value
                
            return inner_dict['keys'], inner_dict['values']
            
    @staticmethod
    def load_hdf5(filename: str) -> dict:
        """Load log dict from HDF5 file.

        Args:
            filename: Path to the HDF5 file to load.

        Returns:
            Dictionary in condensed format (metrics grouped by name).
        """
        log_dict = {}
        for entry_key in StorageManager.read_hdf5_entry_keys(filename):
            keys, values = StorageManager.read_hdf5_entry(filename, entry_key)
            log_dict[entry_key] = {k: v for (k, v) in zip(list(keys), list(values))}
        return log_dict