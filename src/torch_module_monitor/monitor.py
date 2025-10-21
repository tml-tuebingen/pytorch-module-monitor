# Monitor the training of pytorch modules.
import torch
import numpy as np
import math
from typing import Union, List, Tuple, Any, Optional, Callable
from numbers import Number
import logging
import sys
from contextlib import contextmanager
import re

from .hooks import HooksManager
from .storage import StorageManager    


############################################################################################
# Modules can subclass MonitorMixin to implement custom monitoring behavior.
############################################################################################
class MonitorMixin:
    """Modules can subclass this class to log custom metrics during the forward pass (for example, additional statistics of the attention mechanism).

    During the forward pass, call self.is_monitoring() to check if monitoring is active.

    If monitoring is active, call self.get_training_monitor() to get the TrainingMonitor object and log custom metrics.

    The ModuleMonitor checks if modules implement MonitorMixin and automatically calls set_training_monitor on all modules that subclass MonitorMixin.
    """
    def __init__(self):
        self.module_monitor = None

        self.is_reference_module = False 
        """Whether the module is a part of the reference module."""

    def set_module_monitor(self, monitor =None, is_reference_module =False):
        self.module_monitor = monitor
        self.is_reference_module = is_reference_module

    def get_module_monitor(self):
        return self.module_monitor

    @property
    def is_monitoring(self):
        return self.module_monitor is not None and self.module_monitor.is_monitoring()
    

#################################################################
# The ModuleMonitor class
#################################################################
def default_format_module_name_fn(name: str):
    """Default function to format module names for logging.

    Normalizes module names by handling PyTorch compilation wrappers (torch.compile),
    FSDP wrappers, and other internal PyTorch naming conventions.

    Args:
        name: The raw module name from named_modules().

    Returns:
        A formatted module name suitable for logging. Returns "[root module]" for
        the root module and removes wrapper prefixes for all other modules.
    """
    if name == "" or name == "_orig_mod" or name == "_forward_module" or name == "_fsdp_wrapped_module":
        return "[root module]"
    for s in ["_forward_module.", "_orig_mod.", "_fsdp_wrapped_module."]:
        name = name.replace(s, "")
    return name


class ModuleMonitor:
    """Monitor the training of a pytorch module.

    This class can monitor arbitrary statistics of the activations, parameters, and gradients of a pytorch module during training.

    - Activations are logged as "activation/{module_name}/{metric_name}"
    - Parameters are logged as "parameter/{parameter_name}/{metric_name}"
    - Gradients are logged as "gradient/{parameter_name}/{metric_name}"

    The class also supports the comparison of activations and parameters with a so-called "reference module".

    - Activation differences between the module and the reference module are logged as "activation_difference/{module_name}/{metric_name}"
    - Parameter differences are logged as "parameter_difference/{module_name}/{metric_name}"

    For example, the reference module can be the init of the model. Then, we can monitor how the parameters and activations change during training.

    We require that the reference module makes a forward pass with the same input as the monitored module before the monitored module makes a forward pass.

    We support the accumulation of batch statistics over micro-batches. After all micro-batches of a gradient step are done, the function aggregate_step() must be called to aggregate the batch statistics.

    
    """

    def __init__(self, 
                 monitor_step_fn: Optional[Callable[[int], bool]] = lambda step: step % 20 == 0,
                 format_module_name_fn: Callable[[str], str] = default_format_module_name_fn,
                 excluded_modules_regex: str = r'$.^',
                 logger=None,
                 cpu_offload=False): 
        """Init the training monitor."""
        # strategy pattern for the formatting of module names
        self.format_module_name_fn = format_module_name_fn

        # compile the regex for excluding modules
        self.excluded_modules_compiled_re = re.compile(excluded_modules_regex)

        # logging
        if logger is None:
            self.logger = logging.getLogger("ModuleMonitor")
            self.logger.addHandler(logging.NullHandler())  # Silent default logger
        else:
            self.logger = logger
        
        # the module that we monitor
        self.module = None
        self.module_names = {}                                  # a mapping from modules to their names
        self.cpu_offload = cpu_offload                          # whether to offload activations to the CPU

        # the reference module, if any
        self.reference_module = None
        self.ignore_reference_module_activations = False
        self.reference_module_names = {}        # a mapping from modules to their names
        self.reference_module_activations = {}  # a mapping from module names to their activations

        # monitoring state, and the current step
        self.monitor_step_fn = monitor_step_fn
        self.current_step = None
        self.current_step_monitored = False
        self.within_monitoring_window = False  # are we currently monitoring?
        
        # the metrics that we monitor, stored as name -> [regex, metric_fn, metric_aggregation_fn]
        self.activation_metrics = {}
        self.activation_difference_metrics = {}
        self.parameter_metrics = {}
        self.parameter_difference_metrics = {}
        self.gradient_metrics = {}

        # hook managers for the different hooks that we need for monitoring
        self.activation_hooks = HooksManager()
        self.reference_activation_hooks = HooksManager()

        # the storage object where we store the logged metrics
        self.storage = StorageManager(self.logger)


    #################################################################
    # Add metrics to monitor
    #################################################################
    def add_activation_metric(self, metric_name: str, metric_fn: callable, metric_regex: str = ".*", metric_aggregation_fn: Optional[Callable] = None):
        if metric_name in self.activation_metrics:
            raise ValueError(f"Activation metric {metric_name} already exists.")
        
        compiled_re = re.compile(metric_regex)
        self.activation_metrics[metric_name] = (compiled_re, metric_fn, metric_aggregation_fn)

    def add_activation_difference_metric(self, metric_name: str, metric_fn: callable, metric_regex: str = ".*", metric_aggregation_fn: Optional[Callable] = None):
        if metric_name in self.activation_difference_metrics:
            raise ValueError(f"Activation difference metric {metric_name} already exists.")

        compiled_re = re.compile(metric_regex)
        self.activation_difference_metrics[metric_name] = (compiled_re, metric_fn, metric_aggregation_fn)

    def add_parameter_metric(self, metric_name: str, metric_fn: callable, metric_regex: str = ".*"):
        if metric_name in self.parameter_metrics:
            raise ValueError(f"Parameter metric {metric_name} already exists.")

        compiled_re = re.compile(metric_regex)
        self.parameter_metrics[metric_name] = (compiled_re, metric_fn, None)

    def add_parameter_difference_metric(self, metric_name: str, metric_fn: callable, metric_regex: str = ".*"):
        if metric_name in self.parameter_difference_metrics:
            raise ValueError(f"Parameter difference metric {metric_name} already exists.")

        compiled_re = re.compile(metric_regex)
        self.parameter_difference_metrics[metric_name] = (compiled_re, metric_fn, None)

    def add_gradient_metric(self, metric_name: str, metric_fn: callable, metric_regex: str = ".*"):
        if metric_name in self.gradient_metrics:
            raise ValueError(f"Gradient metric {metric_name} already exists.")

        compiled_re = re.compile(metric_regex)
        self.gradient_metrics[metric_name] = (compiled_re, metric_fn, None)


    #################################################################
    # Set the module
    #################################################################
    def set_module(self, module: torch.nn.Module):
        """Set the module that should be monitored.

        We register hooks on the module to implement the monitoring.

        excluded_modules: Excluded from activation monitoring. This is a regex that is matched against the formatted module name. We never register hooks for excluded modules.
        """
        if module is None:
            raise ValueError("Module cannot be None.")

        if self.module is not None: # for simplicity, we do not support changing the module after it has been set
            raise ValueError("Module has already been set!")
        
        # set the module
        self.module = module

        for name, m in module.named_modules():
            # generate the name -> module mapping
            self.module_names[m] = self.format_module_name_fn(name)

            if self._is_excluded(m):
                self.logger.debug(f"Module %s is excluded", name)
                continue

            # register forward hooks for activation monitoring
            hook = self._get_activation_forwad_hook(self.module_names[m]) 
            self.activation_hooks.register_forward_hook(m, hook)
            self.logger.debug(f"Registered forward hook for module %s", name)

            # if the module implements the MonitorMixin interface, set the training monitor
            if isinstance(m, MonitorMixin):
                m.set_module_monitor(self, False)


    #################################################################
    # Add a reference module
    #################################################################
    def set_reference_module(self, module: torch.nn.Module):
        """Set the reference module. 
        
        The training monitor compares the activations and parameters of the monitored module with the reference module. 
        """
        if self.module is None:
            raise ValueError("Please set the monitored module before setting the reference module.")

        # remove any previous reference module
        self.remove_reference_module()

        self.reference_module = module
        self.reference_activation_hooks = HooksManager()
        if module is None:
            return

        # setup for the new reference module
        for name, m in module.named_modules():
            # generate the name -> module mapping
            self.reference_module_names[m] = self.format_module_name_fn(name)

            if self._is_excluded(m, is_reference=True):
                self.logger.debug(f"Excluding reference module %s from monitoring.", name)
                continue

            # register forward hooks for activation monitoring
            hook = self._get_reference_activation_forward_hook()
            self.reference_activation_hooks.register_forward_hook(m, hook)
            self.logger.debug(f"Registered forward hook for reference module %s", name)

            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitorMixin):
                m.set_module_monitor(self, True)

        # check that the reference module names contains the same keys as the monitored module names
        if set(self.module_names.values()) != set(self.reference_module_names.values()):
            raise ValueError("The reference module must have the same structure as the monitored module (there are modules with different names).")


    def has_reference_module(self):
        """Check if a reference module has been set.

        Returns:
            True if a reference module is set, False otherwise.
        """
        return self.reference_module is not None


    def remove_reference_module(self):
        """Remove the reference module and clean up all associated hooks and state."""
        if not self.has_reference_module():
            return
        # notify the MonitoredModules
        for _, m in self.reference_module.named_modules():
            if isinstance(m, MonitorMixin):
                m.set_module_monitor(None, False)
        # remove the reference module
        self.reference_module = None
        self.reference_activation_hooks.remove_all_hooks()
        self.reference_module_names = {}
        self.reference_module_activations = {}


    #################################################################
    # Start step, get metrics, etc.
    # 
    # This is the basic interface used in the training loop
    #################################################################
    def is_step_monitored(self, step: int):
        """Is the given step monitored?"""
        return self.monitor_step_fn(step)
    

    def is_monitoring(self):
        """Are we currently monitoring?"""
        return self.within_monitoring_window


    def begin_step(self, step: int):
        """Notify the monitor that step {step} has started. 
        
        This function should be called before the forward pass.

        Returns True if the step is monitored, False otherwise.
        """
        if self.current_step is not None and step < self.current_step:
            raise ValueError(f"Step {step} cannot be smaller than the current step {self.current_step}.")
        
        # clean-up any previous step (if not done already)
        self.reference_module_activations = {}        

        # do we monitor this step?
        self.current_step = step
        self.current_step_monitored = self.is_step_monitored(step)
        self.within_monitoring_window = self.current_step_monitored

        # check that there is a module to monitor
        if self.current_step_monitored and self.module is None:
            raise ValueError("No module to monitor. Please set the module first.")

        # if we monitor this step, create a new log entry
        if self.current_step_monitored: 
            self.storage.create_step_entry(self.current_step)
            
        return self.current_step_monitored


    @contextmanager
    def no_monitor(self):
        """Context manager to temporarily disable monitoring.

        Use this to perform forward passes that should not be monitored (e.g., validation).

        Example:
            with monitor.no_monitor():
                validation_output = model(validation_input)
        """
        self.within_monitoring_window = False
        try:
            yield
        finally:
            self.within_monitoring_window = self.current_step_monitored


    def after_micro_batch(self):
        """Clean up stored reference activations after processing a micro-batch.

        Call this after each micro-batch when using gradient accumulation, before
        the next micro-batch forward pass.
        """
        self.reference_module_activations = {}


    def end_step(self):
        """Finalize the current step and aggregate metrics across all micro-batches.

        Call this after all micro-batches and backward passes for the step are complete,
        typically before optimizer.step().
        """
        # we don't require the user to call after_micro_batch if there is only a single micro batch
        self.reference_module_activations = {}

        if self.is_monitoring():
            self.storage.aggregate_step(self.current_step)

        self.current_step_monitored = False # we use this to indicate that the step is done
        self.within_monitoring_window = False


    def get_step_metrics(self):
        """Return the metrics logged during the current step.

        Returns:
            Dictionary mapping metric names to their aggregated values for the current step.
        """
        return self.storage.get_step_metrics(self.current_step)
    

    def get_all_metrics(self):
        """Return all logged metrics across all steps.

        Returns:
            Dictionary mapping step numbers to their metric dictionaries.
        """
        return self.storage.get_all_metrics()


    #################################################################
    # Low-level logging functions
    #################################################################

    def log_scalar(self, key: str, value: Union[Number, torch.Tensor], aggregation_fn: Optional[Callable] = None, force=False):
        """Log a scalar value (e.g., loss, learning rate).

        Args:
            key: Name of the metric to log.
            value: Scalar value or single-element tensor.
            aggregation_fn: Function to aggregate values across micro-batches.
            force: If True, log even if current step is not being monitored.
        """
        if not self.is_monitoring() and not force:
            return

        self.storage.log_scalar(self.current_step, key, value, aggregation_fn=aggregation_fn)

    def log_scalars(self, monitor_dict: dict, force=False):
        """Log multiple scalar values at once.

        Args:
            monitor_dict: Dictionary mapping metric names to scalar values.
            force: If True, log even if current step is not being monitored.
        """
        for key, value in monitor_dict.items():
            self.storage.log_scalar(self.current_step, key, value, force=force)

    def log_tensor(self, key: str, tensor: torch.Tensor, aggregation_fn: Optional[Callable] = None):
        """Log a tensor value.

        Args:
            key: Name of the metric to log.
            tensor: Tensor value to log.
            aggregation_fn: Function to aggregate tensors across micro-batches.
        """
        if not self.is_monitoring():
            return

        self.storage.log_tensor(self.current_step, key, tensor, aggregation_fn=aggregation_fn)


    #################################################################
    # Activations
    #################################################################
    def _monitor_activation_metric(self,
                                   module_name :str, 
                                   metric_name :str, 
                                   metric_fn :Callable,
                                   metric_aggregation_fn :Optional[Callable],
                                   activations :torch.Tensor):
        try:
            result = metric_fn(activations)
            log_entry = f"activation/{module_name}/{metric_name}"
            self.log_tensor(log_entry, result, metric_aggregation_fn)
            self.logger.debug(f"Step {self.current_step}: Monitored {metric_name} of activations of module {module_name}, logged as {log_entry} (activation shape {activations.shape}, result shape {result.shape}).")

        except Exception as e:
            raise RuntimeError(f"Failed to monitor metric {metric_name} for the activations of module {module_name} (activation shape {activations.shape}).") from e


    def _monitor_activation_difference_metric(self, 
                                              module_name :str,
                                              metric_name :str,
                                              metric_fn :Callable,
                                              metric_aggregation_fn :Optional[Callable],
                                              activations :torch.Tensor,
                                              ref_activations :torch.Tensor):
        try:
            result = metric_fn(activations, ref_activations)
            log_entry = f"activation_difference/{module_name}/{metric_name}"
            self.log_tensor(log_entry, result, metric_aggregation_fn)
            self.logger.debug(f"Step {self.current_step}: Monitored {metric_name} of activations of module {module_name}, logged as {log_entry} (activation shape {activations.shape}, result shape {result.shape}).")

        except Exception as e:
            raise RuntimeError(f"Failed to monitor metric {metric_name} for the activation differences of module {module_name} (activation shape {activations.shape}).") from e


    def monitor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of a module.

           This function is automatically called by the forward hooks that are registered on all modules of a monitored model.
        
           In addition, this function can be used to monitor activations that are not the output of a module.
           This is what monitor_scaled_dot_product_attention does.
        """
        if not self.is_monitoring() or self._is_excluded(module, is_reference=is_reference):
            return

        try:
            module_name = self._get_module_name(module, is_reference)

            # we only monitor the activations of modules that return a single tensor
            if not isinstance(activations, torch.Tensor):
                self.logger.debug(f"Step {self.current_step}: Ignoring activations of module %s because the model does not return a single tensor (type: %s) (is_reference: %s).", module_name, type(activations), is_reference)
                return

            # detach activations from the graph but keep on the device
            activations = activations.detach().clone()

            # if called with the activations of the reference module, store them in the reference_module_activations dict
            if is_reference:
                if self.ignore_reference_module_activations:
                    return

                # raise a warning if no reference module is set
                if not self.has_reference_module():
                    self.logger.warning(f"Step {self.current_step}: Attempted to store activations of the reference module, but no reference module is set (for module %s).", module_name)
                    return
                # raise an error if the reference module has already stored activations for this module
                if module_name in self.reference_module_activations:
                    raise ValueError(f"Step {self.current_step}: Activations of the reference module for module %s are already stored.", module_name)
                # optionally, offload the activations to the CPU. this is extremely expensive but can save GPU memory.
                if self.cpu_offload:
                    activations = activations.cpu()
                # store the activations
                self.reference_module_activations[module_name] = activations 
                # we are done
                self.logger.debug(f"Step {self.current_step}: Stored activations of reference module %s with shape %s", module_name, activations.shape)
                return

            # activation metrics
            for metric_name, (compiled_re, metric_fn, metric_aggregation_fn) in self.activation_metrics.items():
                if compiled_re.match(module_name):
                    self._monitor_activation_metric(module_name,
                                                    metric_name,
                                                    metric_fn,
                                                    metric_aggregation_fn,
                                                    activations)
                    
            self.logger.debug(f"Step {self.current_step}: Monitored activations of module %s with shape %s", module_name, activations.shape)

            # activation difference metrics
            if self.has_reference_module():
                for metric_name, (compiled_re, metric_fn, metric_aggregation_fn) in self.activation_difference_metrics.items():
                    if compiled_re.match(module_name):
                        if module_name in self.reference_module_activations:
                            ref_activations = self.reference_module_activations[module_name]
                            if self.cpu_offload: # activations need to be on the same device
                                activations = activations.cpu()

                            self._monitor_activation_difference_metric(module_name,
                                                                    metric_name,
                                                                    metric_fn,
                                                                    metric_aggregation_fn,
                                                                    activations,
                                                                    ref_activations)
                        else:
                            self.logger.warning(f"Step {self.current_step}: No reference module activations found for module %s", module_name)

                self.logger.debug(f"Step {self.current_step}: Monitored activation differences of module %s with shape %s", module_name, activations.shape)
        except Exception as e:
            raise RuntimeError(f"Failed to monitor activations of module {module_name}.") from e


    def _get_activation_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.monitor_activations(module_name, output, is_reference=False)
        return hook
    

    def _get_reference_activation_forward_hook(self):
        def hook(module, input, output):
            self.monitor_activations(module, output, is_reference=True)
        return hook


    #################################################################
    # Gradients
    #################################################################
    def monitor_gradients(self, before_clip=False):
        """Compute and log gradient metrics for all parameters.

        Call this after loss.backward() but before optimizer.step().

        Args:
            before_clip: If True, logs metrics as "gradient_before_clip/" instead of "gradient/".
        """
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():
            if param.grad is None:
                self.logger.warning(f"Step {self.current_step}: Found a parameter where the gradient is None: %s", name)
                continue

            # gradient metrics
            for metric_name, (compiled_re, metric_fn, metric_aggregation_fn) in self.gradient_metrics.items():
                if compiled_re.match(name):

                    result = metric_fn(param.grad.detach())

                    # if result is a tensor, apply item()
                    if isinstance(result, torch.Tensor):
                        result = result.item()

                    # Create log entry
                    log_entry = f"gradient/{self.format_module_name_fn(name)}/{metric_name}"
                    if before_clip:
                        log_entry = log_entry.replace("gradient/", "gradient_before_clip/", 1)
                    self.log_scalar(log_entry, result)
            
            self.logger.debug(f"Step {self.current_step}: Monitored gradient of parameter %s with shape %s with %s %s (logged as %s)", name, param.grad.shape, metric_name, result, log_entry)


    #################################################################
    # Parameters
    #################################################################
    def _monitor_parameter(self, name, param, metric_fn, metric_name):
        """Low-level function that logs a metric for a parameter."""
        try:
            result = metric_fn(param)

            # if result is a tensor, apply item()
            if isinstance(result, torch.Tensor):
                result = result.item()

            # Create log entry
            log_entry = f"parameter/{self.format_module_name_fn(name)}/{metric_name}"
            self.log_scalar(log_entry, result)
            self.logger.debug(f"Step {self.current_step}: Monitored parameter %s with shape %s with %s %s (logged as %s)", name, param.shape, metric_name, result, log_entry)

        except Exception as e:
            raise RuntimeError(f"Failed to monitor parameter {name} of shape {param.shape} with metric {metric_name}.") from e


    def _monitor_parameter_difference(self, name, param, ref_param, metric_fn, metric_name):
        """Low-level function that logs a metric for the difference between a parameter and its reference."""
        try:
            result = metric_fn(param, ref_param)

            # if result is a tensor, apply item()
            if isinstance(result, torch.Tensor):
                result = result.item()

            # Create log entry
            log_entry = f"parameter_difference/{self.format_module_name_fn(name)}/{metric_name}"
            self.log_scalar(log_entry, result)
            self.logger.debug(f"Step {self.current_step}: Monitored parameter difference %s with shape %s with %s %s (logged as %s)", name, param.shape, metric_name, result, log_entry)

        except Exception as e:
            raise RuntimeError(f"Failed to monitor parameter difference {name} of shape {param.shape} with metric {metric_name}.") from e


    def monitor_parameters(self):
        """Compute and log parameter metrics for all parameters.

        Call this during the training step to monitor parameter values and,
        if a reference module is set, parameter differences.
        """
        if not self.is_monitoring():
            return
        
        # collect the parameters of the reference module
        reference_module_parameters = {}
        if self.reference_module is not None and len(self.parameter_difference_metrics) > 0:
            for name, param in self.reference_module.named_parameters():
                reference_module_parameters[name] = param
        
        for name, param in self.module.named_parameters():
            # parameter metrics
            for metric_name, (compiled_re, metric_fn, metric_aggregation_fn) in self.parameter_metrics.items():
                if compiled_re.match(name):
                    self._monitor_parameter(name, param, metric_fn, metric_name)

            # parameter difference metrics
            if self.reference_module is not None:
                if name in reference_module_parameters:
                    ref_param = reference_module_parameters[name]
                        
                    for metric_name, (compiled_re, metric_fn, metric_aggregation_fn) in self.parameter_difference_metrics.items():
                        if compiled_re.match(name):
                            self._monitor_parameter_difference(name, param, ref_param, metric_fn, metric_name)
                else:
                    self.logger.warning(f"Step {self.current_step}: Parameter %s not found in the reference module.", name)


    #################################################################
    # Additional internals
    #################################################################
    def _get_module_name(self, module :Union[str, torch.nn.Module], is_reference :bool) -> str:
        """Look up the name of a torch.nn.Module in the appropriate dict."""
        module_name = module
        if isinstance(module_name, str): # nothing to do if we already have a string
            return module_name
        # otherwise, look up in the respective dict
        if is_reference:
            if not module in self.reference_module_names:
                raise ValueError(f"Module {module} not found in reference module names.")
            module_name = self.reference_module_names[module]
        else:
            if not module in self.module_names:
                raise ValueError(f"Module {module} not found in module names.")
            module_name = self.module_names[module]
        return module_name
    
    
    def _is_excluded(self, module :Union[str, torch.nn.Module], is_reference=False) -> bool:
        """Check if a module is excluded from monitoring."""
        module_name = self._get_module_name(module, is_reference=is_reference)
        return self.excluded_modules_compiled_re.match(module_name) is not None
    
