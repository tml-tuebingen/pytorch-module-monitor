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

from hooks import ModuleHooksManager
from storage import StorageManager    

############################################################################################
# Modules can subclass MonitoredModule to implement custom monitoring behavior.
############################################################################################
class MonitorMixin:
    """A torch.nn.Module can subclass this class to log custom metrics during the forward pass.
    This is used to monitor the attention operation.

    During the foward pass, the module can obtain the training monitor via get_training_monitor(). 

    TrainingMonitor automatically calls set_training_monitor on all modules that subclass MonitoredModule.
    """
    def __init__(self):
        self.training_monitor = None

        self.is_reference_module = False 
        """Whether the module is a part of the reference module."""

    def set_training_monitor(self, monitor =None, is_reference_module =False):
        self.training_monitor = monitor
        self.is_reference_module = is_reference_module

    def get_training_monitor(self):
        return self.training_monitor

    @property
    def is_monitoring(self):
        return self.training_monitor is not None and self.training_monitor.is_monitoring()
    


#################################################################
# The ModuleMonitor class
#################################################################
class ModuleMonitor:
    """Monitor the training of a pytorch module.

    This class can monitor arbitrary statistics of the activations, parameters, and gradients of a pytorch module during training.

    - Activations are logged as "activations/{module_name}/{metric_name}"
    - Parameters are logged as "parameters/{parameter_name}/{metric_name}"
    - Gradients are logged as "gradients/{parameter_name}/{metric_name}"

    The class also supports the comparison of activations and parameters with a so-called "reference module".

    - Activation differences between the module and the reference module are logged as "activation_differences/{module_name}/{metric_name}"
    - Parameter differences are logged as "parameter_differences/{module_name}/{metric_name}"

    For example, the reference module can be the init of the model. Then, we can monitor how the parameters and activations change during training.

    We require that the reference module makes a forward pass with the same input as the monitored module before the monitored module makes a forward pass.

    We support the accumulation of batch statistics over micro-batches. After all micro-batches of a gradient step are done, the function aggregate_step() must be called to aggregate the batch statistics.

    
    """
    
    #################################################################
    # Setup
    #################################################################
    def __init__(self, 
                 module = None,
                 start_step = 0,
                 monitor_interval = 20,
                 format_module_name_fn: Optional[Callable[[str], str]] = None,
                 logger=None,
                 cpu_offload=False): 
        """Init the training monitor."""
        self.format_module_name_fn = format_module_name_fn


        self.module = None
        self.module_hooks = {}
        self.activation_difference_hooks = {}   # used by monitor_activation_differences
        self.module_names = {}                  # a mapping from modules to their names
        self.cpu_offload = cpu_offload          # whether to offload activations to the CPU

        self.reference_module = None
        self.reference_module_hooks = None
        self.ignore_reference_module_activations = False
        self.reference_module_names = {}        # a mapping from modules to their names
        self.reference_module_activations = {}  # a mapping from module names to their activations

        self.monitor_interval = monitor_interval
        self.step = start_step
        self.monitor = monitor
        self.monitor_step = False # do we monitor the current gradient step?

        # the metrics that we monitor, stored as name -> [regex, metric_fn]
        self.activation_metrics = {}
        self.activation_difference_metrics = {}
        self.parameter_metrics = {}
        self.parameter_difference_metrics = {}
        self.gradient_metrics = {}

        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger("ModuleMonitor")
            self.logger.addHandler(logging.NullHandler())  # Silent default logger
        else:
            self.logger = logger

        # here we store the logged metrics
        self.storage = StorageManager(self.logger)

        if module is not None:
            self.set_module(module)




    #################################################################
    # Set the module
    #################################################################
    def set_module(self, module):
        """Set the module that we want to monitor. This function will register forward hooks on the module to monitor the activations. It will also remove any previously set module and reference module."""
        if not self.monitor:
            self.logger.debug(f"Step {self.step}: Monitoring is disabled. Not setting module.")
            return
        
        # remove any previous reference module
        self.remove_reference_module()
        
        # remove any previous module
        if self.module is not None:
            self.module = None
            for hook in self.module_hooks.values(): # remove all hooks
                hook.remove()
            self.module_hooks = {}
            for hook in self.activation_difference_hooks.values():
                hook.remove()
            self.activation_difference_hooks = {}
            self.module_names = {}
            # if the module implements the MonitoredModule interface, remove the training monitor
            for _, m in module.named_modules():
                if isinstance(m, MonitorMixin):
                    m.set_training_monitor(None, False)

        # setup for the new module
        self.module = module
        for name, m in module.named_modules():
            self.module_names[m] = format_module_name(name)
            self.module_hooks[m] = m.register_forward_hook(self._get_activation_forwad_hook(self.module_names[m]))
            self.logger.debug(f"Step {self.step}: Registered forward hook for module %s", name)
            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitorMixin):
                m.set_training_monitor(self, False)

    #################################################################
    # Add a reference module
    #################################################################
    def set_reference_module(self, module):
        """Set the reference module. The training monitor compares the activations and parameters of the monitored module with the reference module. 
        The reference module must take a forward pass with the same input as the monitored module BEFORE the monitored module takes a forward pass."""
        if not self.monitor:
            self.logger.debug(f"Step {self.step}: Monitoring is disabled. Not setting reference module.")
            return
        
        # remove any previous reference module
        self.remove_reference_module()

        self.reference_module = module
        if module is None:
            return
        
        # setup for the new reference module
        # register the modules via their names and set forward hooks
        for name, m in module.named_modules():
            self.reference_module_names[m] = self._format_module_name(name)
            self.reference_module_hooks[m] = m.register_forward_hook(self._get_reference_activation_forwad_hook())

            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitorMixin):
                m.set_training_monitor(self, True)

        # assert that the reference module names contains the same keys as the monitored module names
        assert set(self.module_names.values()) == set(self.reference_module_names.values()), "The reference module must have the same structure as the monitored module (there are modules with different names)."


    def remove_reference_module(self):
        """Remove the reference module."""
        if not self.has_reference_module():
            return
        # notify the MonitoredModules
        for _, m in self.reference_module.named_modules():
            if isinstance(m, MonitorMixin):
                m.set_training_monitor(None, False)
        # remove the reference module
        self.reference_module = None
        for hook in self.reference_module_hooks.values():
            hook.remove()
        self.reference_module_hooks = {}
        self.reference_module_names = {}
        self.reference_module_activations = {}


    def has_reference_module(self): 
        return self.reference_module is not None


    #################################################################
    # Specify what to monitor
    #################################################################
    def add_activation_metric(self, metric_name: str, metric_fn: callable, metric_regex: str):
        if metric_name in self.activation_metrics:
            raise ValueError(f"Activation metric {metric_name} already exists.")
        
        compiled_re = re.compile(metric_regex)
        self.activation_metrics[metric_name] = (compiled_re, metric_fn)

    def add_activation_difference_metric(self, metric_name: str, metric_fn: callable, metric_regex: str):
        if metric_name in self.activation_difference_metrics:
            raise ValueError(f"Activation difference metric {metric_name} already exists.")

        compiled_re = re.compile(metric_regex)
        self.activation_difference_metrics[metric_name] = (compiled_re, metric_fn)


    #################################################################
    # Start step, get metrics, etc.
    # 
    # This is the interface used in the training loop
    #################################################################
    def start_step(self, step: int):
        """Notify the monitor that a new step has started. This function should be called before the forward pass. Returns True if the step should be monitored, False otherwise."""
        # clean-up any previous step (if not done already)
        self.reference_module_activations = {}        

        # do we monitor this step?
        self.monitor_step = self.is_step_monitored(step)
        self.step = step

        # check that there is a module to monitor
        if self.monitor_step and self.module is None:
            raise ValueError("No module to monitor. Please set the module first.")

        # if we monitor this step, create a new entry in the log dict
        if self.monitor_step: 
            self.log_dict[step] = {}
            
        return self.monitor_step
        

    def is_step_monitored(self, step: int):
        """Will we monitor the given step? Useful to check if a future step will be monitored."""
        if not self.monitor: # global toggle to turn off monitoring
            return False
        monitor_step = step % self.monitor_interval == 1
        if step <= 20:  # we always monitor the first 20 steps
            monitor_step = True
        if not monitor_step and step <= 100: # more frequent monitoring for the first 100 steps
            monitor_step = step % 20 == 1
        return monitor_step
    

    def is_monitoring(self):
        return self.monitor_step
    

    @contextmanager
    def no_monitor(self):
        """
        Context manager to temporarily disable all monitoring. Use this to compute the validation loss and perform other forward operations that should not be monitored.
        """
        original_state = self.monitor_step
        self.monitor_step = False
        try:
            yield
        finally:
            self.monitor_step = original_state


    def after_micro_batch(self):
        """To be called after a mini-batch is done. Cleanup of intermediate state."""
        self.reference_module_activations = {}
    

    def aggregate_step(self):
        """This function is called after all mini-batches of a gradient step are done. It aggregates the batch statistics. """
        if self.is_monitoring():
            self.storage.aggregate_step()
    

    def get_step_metrics(self):
        """Return the log dict of the current step"""
        if self.monitor_step:
            return self.log_dict[self.step]
        return {}
    

    def get_all_metrics(self):
        """Return the full log dict with all steps that have been logged so far."""
        return self.log_dict
    

    def load_metrics(self, log_dict):
        """Load a log dict."""
        # assert that the current log dict is empty
        if len(self.log_dict) > 0:
            raise ValueError("The current log dict is not empty. Please clear it first.")
        self.log_dict = log_dict


    #################################################################
    # Low-level logging functions
    #################################################################
    
    def log_scalar(self, key: str, value: Union[Number, torch.Tensor], force=False):
        """Monitor a scalar value such as the loss or the learning rate.
        
        If force is set to True, the value is logged even if we are not monitoring the current step.
        This is useful to monitor values like the final validation loss.
        """
        if not self.is_monitoring() and not force:
            return
        
        self.storage.log_scalar(self.step, key, value)

    def log_scalars(self, monitor_dict: dict, force=False):
        """Monitor a dictionary of scalar values."""
        for key, value in monitor_dict.items():
            self.storage.log_scalar(self.step, key, value, force=force)

    def log_tensor(self, key: str, tensor: torch.Tensor):
        """Monitor a torch.Tensor."""
        if not self.is_monitoring():
            return
        
        self.storage.log_tensor(self.step, key, tensor)


    #################################################################
    # Activations
    #################################################################
    def monitor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of a module.

           This function is automatically called by the forward hooks that are registered on all modules of a monitored model.
        
           In addition, this function can be used to monitor activations that are not the output of a module.
           This is what monitor_scaled_dot_product_attention does.
        """
        if not self.is_monitoring():
            return

        # assert that module_name is a string
        module_name = self._module_name(module, is_reference)

        # detach activations from the graph but keep on the device
        activations = activations.detach()

        # if called with the activations of the reference module, store them in the reference_module_activations dict
        if is_reference:
            if self.ignore_reference_module_activations:
                return

            # raise a warning if no reference module is set
            if not self.has_reference_module():
                self.logger.warning(f"Step {self.step}: Attempted to monitor activations of the reference module, but no reference module is set (for module %s).", module_name)
                return
            # raise a warning if the reference module has already stored activations for this module
            if module_name in self.reference_module_activations:
                self.logger.warning(f"Step {self.step}: Attempted to monitor activations of the reference module for module %s, but activations are already stored.", module_name)
                return
            # optionally, offoad the activations to the CPU. this is extremely expensive but saves GPU memory.
            if self.cpu_offload:
                activations = activations.cpu()
            # store the activations
            self.reference_module_activations[module_name] = activations 
            # we are done
            return

        # monitor the different activation metrics
        for metric_name, metric_fn in self.activation_metrics.items():
            # compute the metric
            result = metric_fn(activations)

            # monitor the metric
            log_entry = f"activation/{module_name}/{metric_name}"
            self.log_tensor(log_entry, result)

        # metrics based on the activations and the reference activations (most likely L2 norm)
        if self.has_reference_module():
            if module_name in self.reference_module_activations:
                ref_activations = self.reference_module_activations[module_name]
                if self.cpu_offload: # activations need to be on the same device
                    activations = activations.cpu()

                for metric_name, metric_fn in self.activation_difference_metrics.items():
                    # compute the metric
                    result = metric_fn(activations, ref_activations)

                    log_entry = f"activation_difference/{module_name}/{metric_name}"
                    self.log_tensor(log_entry, result)
            else:
                self.logger.warning(f"Step {self.step}: No reference module activations found for module %s", module_name)

        self.logger.debug(f"Step {self.step}: Monitored activations of module %s with shape %s", module_name, activations.shape)


    def _get_activation_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.monitor_activations(module_name, output, is_reference=False)
        return hook
    

    def _get_reference_activation_forwad_hook(self):
        def hook(module, input, output):
            self.monitor_activations(module, output, is_reference=True)
        return hook
    

    #################################################################
    # Gradients
    #################################################################
    def monitor_gradients(self, before_clip=False):
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():
            if param.grad is None:
                self.logger.warning(f"Step {self.step}: Found a parameter where the gradient is None: %s", name)
                continue

            # log the different metrics (most likely the frobenius norm of the gradients)
            for metric_name, metric_fn in self.gradient_metrics.items():
                # we apply the metrics to the flattened gradient tensor
                result = metric_fn(param.grad.detach().flatten())

                # if result is a tensor, apply item()
                if isinstance(result, torch.Tensor):
                    result = result.item()

                # Create log entry
                log_entry = f"gradient/{self._format_module_name(name)}/{metric_name}"
                if before_clip:
                    log_entry = log_entry.replace("gradient/", "gradient_before_clip/", 1)

                self.log_scalar(log_entry, result)
                self.logger.debug(f"Step {self.step}: Monitored gradient of parameter %s with shape %s with %s %s (logged as %s)", name, param.grad.shape, metric_name, result, log_entry)


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
            log_entry = f"parameter/{self._format_module_name(name)}/{metric_name}"
            self.log_scalar(log_entry, result)
            self.logger.debug(f"Step {self.step}: Monitored parameter %s with shape %s with %s %s (logged as %s)", name, param.shape, metric_name, result, log_entry)

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
            log_entry = f"parameter_difference/{self._format_module_name(name)}/{metric_name}"
            self.log_scalar(log_entry, result)
            self.logger.debug(f"Step {self.step}: Monitored parameter difference %s with shape %s with %s %s (logged as %s)", name, param.shape, metric_name, result, log_entry)

        except Exception as e:
            raise RuntimeError(f"Failed to monitor parameter difference {name} of shape {param.shape} with metric {metric_name}.") from e


    def monitor_parameters(self):
        """Monitor the parameters of the monitored module."""
        if not self.is_monitoring():
            return
        
        # collect the parameters of the reference module
        reference_module_parameters = {}
        if self.reference_module is not None and len(self.parameter_difference_metrics_spec) > 0:
            for name, param in self.reference_module.named_parameters():
                reference_module_parameters[name] = param
        
        for name, param in self.module.named_parameters():
            # log metrics that apply to all parameters
            for metric_name, metric_fn in self.parameter_metrics.items():
                self._monitor_parameter(name, param, metric_fn, metric_name)
                
            # log metrics that specify via a regular expression that they only apply to specific named parameters
            for compiled_re, metrics_dict in self.parameter_metrics_spec.items():
                if compiled_re.match(name):
                    for metric_name, metric_fn in metrics_dict.items():
                        self._monitor_parameter(name, param, metric_fn, metric_name) # we put this in a try-catch block because it executes 

            # log metrics for the difference between the parameter and its reference
            if self.reference_module is not None:
                if name in reference_module_parameters:
                    ref_param = reference_module_parameters[name]
                        
                    # log metrics that specify via a regular expression that they only apply to specific named parameters
                    for compiled_re, metrics_dict in self.parameter_difference_metrics_spec.items():
                        if compiled_re.match(name):
                            for metric_name, metric_fn in metrics_dict.items():
                                self._monitor_parameter_difference(name, param, ref_param, metric_fn, metric_name)
                else:
                    self.logger.warning(f"Step {self.step}: Parameter %s not found in the reference module.", name)


    #################################################################
    # Internal helper functions
    #################################################################
    def _module_name(self, module :Union[str, torch.nn.Module], is_reference :bool) -> str:
        """Look up the name of a torch.nn.Module in the appropriate dict."""
        module_name = module
        if isinstance(module_name, str): # when the module name is already provided as a string
            return module_name
        # handle normal and reference modules
        if is_reference:
            module_name = self.reference_module_names[module]
        else:
            if not module in self.module_names:
                self.logger.warning(f"Step {self.step}: Module %s not found in the module names dict.", module)
                return "[unknown module]"
            module_name = self.module_names[module]
        assert isinstance(module_name, str)
        return module_name
    
    def _format_module_name(self, name: str):
        return self.format_module_name_fn(name) if self.format_module_name_fn is not None else ("[root module]" if name == "" else name)