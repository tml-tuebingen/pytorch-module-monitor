import re
from typing import Dict, Callable, Optional, Union, Any
import torch.nn as nn


class ModuleHooksManager:
    """Manages forward hooks for a module and its submodules with automatic cleanup."""
    
    def __init__(self, module: nn.Module, format_module_name_fn: Optional[Callable[[str], str]] = None):
        """Initialize the ModuleHooksManager.
        
        Args:
            module: The root module to manage hooks for
            format_module_name_fn: Optional function to format module names.
                                  Takes a module name string and returns formatted string.
                                  If None, uses default formatting (only handles root module).
        """
        self.module = module
        self.hooks: Dict[nn.Module, Any] = {}
        self.module_names: Dict[nn.Module, str] = {}
        
        # Use custom formatter or default
        if format_module_name_fn is None:
            format_module_name_fn = lambda name: "[root module]" if name == "" else name
        
        # Register all submodule names
        for name, submodule in module.named_modules():
            self.module_names[submodule] = format_module_name_fn(name)
    
    def register_forward_hook(self, 
                              hook_fn: Callable,
                              target: Optional[Union[str, re.Pattern]] = None) -> None:
        """Register a forward hook on specified modules.
        
        Args:
            hook_fn: The hook function to register
            target: Can be:
                - None: register on all modules
                - str: register on modules with matching name (substring match)
                - re.Pattern: register on modules matching regex
        """
        modules = self._get_target_modules(target)
        
        for module in modules:
            # Remove existing hook if present
            if module in self.hooks:
                self.hooks[module].remove()
            
            handle = module.register_forward_hook(hook_fn)
            self.hooks[module] = handle
    
    def _get_target_modules(self, target: Optional[Union[str, re.Pattern]]) -> list:
        """Get list of modules based on target specification."""
        if target is None:
            # All modules
            return list(self.module_names.keys())
        
        elif isinstance(target, str):
            # String substring match
            return [m for m, name in self.module_names.items() if target in name]
        
        elif isinstance(target, re.Pattern):
            # Regex match
            return [m for m, name in self.module_names.items() if target.match(name)]
        
        else:
            raise ValueError(f"Invalid target type: {type(target)}. Expected None, str, or re.Pattern")
    
    def remove(self, target: Optional[Union[str, re.Pattern]] = None) -> None:
        """Remove hooks from specified modules.
        
        Args:
            target: Same as register method
        """
        if target is None:
            # Remove all hooks
            for handle in self.hooks.values():
                handle.remove()
            self.hooks.clear()
        else:
            # Remove hooks from specific modules
            modules = self._get_target_modules(target)
            for module in modules:
                if module in self.hooks:
                    self.hooks[module].remove()
                    del self.hooks[module]
    
    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def get_module_name(self, module: nn.Module) -> Optional[str]:
        """Get the registered name of a module."""
        return self.module_names.get(module)
    
    def __len__(self) -> int:
        """Get count of registered hooks."""
        return len(self.hooks)
    
    def __del__(self):
        """Cleanup: remove all hooks when object is destroyed."""
        self.remove_all()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.remove_all()


# Example usage:
"""
# Create hooks manager with default name formatting
hooks = ModuleHooksManager(model)

# Create hooks manager with custom name formatting (e.g., for FSDP)
def format_fsdp_names(name: str) -> str:
    if name == "" or name == "_orig_mod" or name == "_forward_module":
        return "[root module]"
    for s in ["_forward_module.", "_orig_mod.", "_fsdp_wrapped_module."]:
        name = name.replace(s, "")
    return name

hooks = ModuleHooksManager(model, format_module_name_fn=format_fsdp_names)

# Register hook on all modules
hooks.register_forward_hook(my_hook_function)

# Register hook on modules containing "attention" in the name
hooks.register_forward_hook(attention_hook, target="attention")

# Register hook using regex pattern
import re
linear_pattern = re.compile(r".*\.linear\d+")
hooks.register_forward_hook(linear_hook, target=linear_pattern)

# Remove specific hooks
hooks.remove(target="attention")

# Remove all hooks
hooks.remove_all()

# Get number of registered hooks
print(f"Registered {len(hooks)} hooks")

# Automatic cleanup with context manager
with ModuleHooksManager(model) as hooks:
    hooks.register_forward_hook(my_hook)
    # ... do something
# hooks automatically removed here
"""