#######################################################################################
# Perform the refined muP coordinate check (RCC) from https://arxiv.org/abs/2505.22491
#######################################################################################
import torch
from typing import Union, List, Tuple, Any

from .hooks import HooksManager
from .monitor import ModuleMonitor  

def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm along last dimension."""
    return torch.linalg.vector_norm(tensor, ord=2, dim=-1)

class RefinedCoordinateCheck:
    """Performs the *refined* coordinate check (RCC) from https://arxiv.org/abs/2505.22491
    """

    #######################################################################################
    # Public interface
    #######################################################################################
    def __init__(self,
                 monitor: ModuleMonitor):
        if monitor.module is None:
            raise ValueError("Set the module to monitor before setting up the refined coordinate check.")
        if monitor.reference_module is None:
            raise ValueError("Set the reference module before setting up the refined coordinate check.")

        self.monitor = monitor

        # copy some key attributes from the monitor
        self.logger = monitor.logger
        self.module = monitor.module
        self.reference_module = monitor.reference_module
        self.format_module_name_fn = monitor.format_module_name_fn
        self.cpu_offload = monitor.cpu_offload

        self.module_inputs = {}             # a mapping from module names to their inputs
        self.module_outputs = {}            # a mapping from module names to their outputs (activations)
        self.reference_module_inputs = {}   # a mapping from module names to their inputs, but for the reference module
        self.rcc_forward_hooks = HooksManager()
        self.rcc_reference_forward_hooks = HooksManager()

        # register forward hooks on the module
        for name, module in self.monitor.module.named_modules():
            name = self.monitor.format_module_name_fn(name)
            hook = self._get_rcc_forward_hook(name)
            self.rcc_forward_hooks.register_forward_hook(module, hook)
            self.logger.debug(f"Registered rcc forward hook for module %s", name)

        # register forward hooks on the reference module
        for name, module in self.monitor.reference_module.named_modules():
            name = self.monitor.format_module_name_fn(name)
            hook = self._get_rcc_reference_forward_hook(name)
            self.rcc_reference_forward_hooks.register_forward_hook(module, hook)
            self.logger.debug(f"Registered rcc forward hook for reference module %s", name)


    def refined_coordinate_check(self):
        """Perform the *refined* coordinate check (RCC) from https://arxiv.org/abs/2505.22491.

        This function should be called after the forward pass of the monitored module.

        This function performs additional forward passes. It compares the activation differences of the monitored module and the reference module.
        """
        if not self.monitor.is_monitoring():
            return

        if len(self.module_inputs) == 0:
            raise ValueError("No inputs found for the monitored module. Perform a forward pass before calling refined_coordinate_check.")
        
        # we assume all t operations take place of the device of the monitored module
        device = next(self.module.parameters()).device

        # iterate over module to perform coordinate check
        comparison_modules = dict(self.reference_module.named_modules())

        for name, module in self.module.named_modules():
            if not self.format_module_name_fn(name) in self.module_inputs: # we only perform the coordinate check for modules that run "forward" / call their forward hooks (this excludes ModuleList / ModuleDict)
                continue
            if self.format_module_name_fn(name) == "[root module]": # we exclude the root module (same result as regular reference module forward pass)
                continue

            # TODO we can make this code not fail / only provide a warning if a key is not in the dict (that is, a forward hooks was not called)
            comparison_module = comparison_modules[name]
            name = self.format_module_name_fn(name)
            comparison_input = self.reference_module_inputs[name]
            comparison_output = self.monitor.reference_module_activations[name]
            module_input = self.module_inputs[name]
            module_output = self.module_outputs[name]

            # move all tensors to the specified device
            comparison_input = tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in comparison_input)
            comparison_output = comparison_output.to(device)
            module_input = tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in module_input)
            module_output = module_output.to(device)
            
            self._module_refined_coordinate_check(name, 
                                              module_input, 
                                              module_output, 
                                              module,
                                              comparison_input,
                                              comparison_output, 
                                              comparison_module)
            
        # clear stored inputs and outputs
        self.module_inputs = {}             
        self.module_outputs = {}
        self.reference_module_inputs = {}


    #######################################################################################
    # Implementation details
    #######################################################################################
    def _module_refined_coordinate_check(self, 
                                     module_name: str, 
                                     Wt_input: Tuple[Any],            
                                     Wt_xt: torch.Tensor, 
                                     Wt_module: torch.nn.Module,        
                                     W0_input: Tuple[Any],              
                                     W0_x0: torch.Tensor,       
                                     W0_module: torch.nn.Module):
            
            self.logger.debug(f"Step {self.monitor.current_step}: Performing refined coordinate check for module %s with shape %s ...", module_name, Wt_xt.shape)
            
            with torch.no_grad():
                # perform a forward pass in the reference module, using the intermediate input from the monitored module (W_0 x_t)
                self.monitor.ignore_reference_module_activations = True      # temporarily ignore reference module activation hooks (this is important! we do not want to overwrite the reference module activations)
                W0_xt = W0_module(*Wt_input).detach()
                self.monitor.ignore_reference_module_activations = False

                # for modules that have a .bias attribute, additionally compute the metrics without the bias
                W0_x0_nobias, Wt_xt_nobias, W0_xt_nobias = None, None, None
                if hasattr(Wt_module, "bias"):
                    if Wt_module.bias is None: # no bias for this module, we are already done
                        W0_x0_nobias = W0_x0
                        W0_xt_nobias = W0_xt
                        Wt_xt_nobias = Wt_xt
                    else: # substract bias
                        try:
                            W0_x0_nobias = W0_x0 - W0_module.bias
                            W0_xt_nobias = W0_xt - W0_module.bias
                            Wt_xt_nobias = Wt_xt - Wt_module.bias
                        except Exception as e:
                            self.logger.warning(f"Step {self.monitor.current_step}: Refined coordinate check: Failed to compute bias-free activations for module %s: %s", module_name, e)
                            W0_x0_nobias, W0_xt_nobias, Wt_xt_nobias = None, None, None

            if isinstance(Wt_module, torch.nn.Embedding):      # special handling for embedding layers: we only compute (W_t-W_0) x_t           
                result = l2_norm(Wt_xt - W0_xt)
                log_entry = f"RCC (W_t-W_0)x_t/{module_name}/l2norm"
                self.monitor.log_tensor(log_entry, result)
                return

            # Frobenious norm of (W_t-W_0) x_t            
            result = l2_norm(Wt_xt - W0_xt)
            log_entry = f"RCC (W_t-W_0)x_t/{module_name}/l2norm"
            self.monitor.log_tensor(log_entry, result)

            # Frobenious norm of W_0 (x_t-x_0)
            result = l2_norm(W0_xt - W0_x0)
            log_entry = f"RCC W_0(x_t-x_0)/{module_name}/l2norm"
            self.monitor.log_tensor(log_entry, result)

            # norm of x_t
            xt = Wt_input[0]
            if isinstance(xt, torch.Tensor):
                self.logger.debug(f"Step {self.monitor.current_step}: Refined coordinate check: Input to module {module_name} is a tensor with shape {xt.shape}")
                result = l2_norm(xt)
                log_entry = f"RCC x_t/{module_name}/l2norm"
                self.monitor.log_tensor(log_entry, result)
            else:
                self.logger.debug(f"Step {self.monitor.current_step}: Refined coordinate check: Input to module {module_name} is not a tensor, but {type(xt)}")

            # norm of x_t - x_0
            x0 = W0_input[0]
            if isinstance(x0, torch.Tensor) and isinstance(xt, torch.Tensor):
                result = l2_norm(xt - x0)
                log_entry = f"RCC (x_t-x_0)/{module_name}/l2norm"
                self.monitor.log_tensor(log_entry, result)
            else:
                self.logger.debug(f"Step {self.monitor.current_step}: Refined coordinate check: Input to module {module_name} is not a tensor, but {type(x0)}")

            # for linear layers and layer norm layers, the terms without the bias provide to the coordinate check for the weight
            if isinstance(Wt_module, torch.nn.Linear) or isinstance(Wt_module, torch.nn.LayerNorm):
                result = l2_norm(Wt_xt_nobias - W0_xt_nobias)       # Bias free Frobenious norm of (W_t-W_0) x_t
                log_entry = f"RCC (W_t-W_0)x_t/{module_name}.weight/l2norm"
                self.monitor.log_tensor(log_entry, result)

                result = l2_norm(W0_xt_nobias - W0_x0_nobias)       # Bias free Frobenious norm of W_0 (x_t-x_0)
                log_entry = f"RCC W_0(x_t-x_0)/{module_name}.weight/l2norm"
                self.monitor.log_tensor(log_entry, result)

                result = l2_norm(Wt_xt_nobias)                      # bias free norm of Wt_xt
                log_entry = f"RCC (W_t-W_0)x_t/{module_name}.weight/W_t x_t/l2norm"
                self.monitor.log_tensor(log_entry, result)

                result = l2_norm(W0_x0_nobias)                      # bias free norm of W0_x0
                log_entry = f"RCC W_0(x_t-x_0)/{module_name}.weight/W_0 x_0/l2norm"
                self.monitor.log_tensor(log_entry, result)

            # for layer norm, additionally log x-E(x)/Var(x) as input to the weights
            if isinstance(Wt_module, torch.nn.LayerNorm):           
                xt = torch.nn.functional.layer_norm(Wt_input[0], Wt_module.normalized_shape, None, None, Wt_module.eps)
                result = l2_norm(xt)
                log_entry = f"RCC (W_t-W_0)x_t/{module_name}.weight/x_t/l2norm"
                self.monitor.log_tensor(log_entry, result)

                x0 = torch.nn.functional.layer_norm(W0_input[0], W0_module.normalized_shape, None, None, W0_module.eps)
                result = l2_norm(xt - x0)
                log_entry = f"RCC W_0(x_t-x_0)/{module_name}.weight/x_t-x_0/l2norm"
                self.monitor.log_tensor(log_entry, result)


    def _get_rcc_forward_hook(self, module_name: str):
        def hook(module, input, output):
            if not self.monitor.is_monitoring():
                return
            
            # detach input and output from the computational graph
            input = tuple(i.detach() if isinstance(i, torch.Tensor) else i for i in input)
            output = output.detach()

            # move the input and output to the CPU if cpu_offload is set
            if self.cpu_offload:
                input = (i.cpu() if isinstance(i, torch.Tensor) else i for i in input)
                output = output.cpu()

            # store the input and output
            self.module_inputs[module_name] = input
            self.module_outputs[module_name] = output
        return hook
    

    def _get_rcc_reference_forward_hook(self, module_name: str):
        def hook(module, input, output):
            if not self.monitor.is_monitoring():
                return
            
            if self.monitor.ignore_reference_module_activations: # very important! otherwise we re-set the inputs to the reference modules during the additional rcc-forward passes
                                                         # in the future, this should be replaced with a more general no_monitor context manager
                return
            
            # same as for _get_rcc_forward_hook, but we already have the activations, so we only need to store the inputs
            input = tuple(i.detach() if isinstance(i, torch.Tensor) else i for i in input)

            # move the input and output to the CPU if cpu_offload is set
            if self.cpu_offload:
                input = (i.cpu() if isinstance(i, torch.Tensor) else i for i in input)

            # store the input and output
            self.reference_module_inputs[module_name] = input
        return hook
    

