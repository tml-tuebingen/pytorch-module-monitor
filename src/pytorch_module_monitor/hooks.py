import re
from typing import Dict, Callable, Optional, Union, Any
import torch.nn as nn


class HooksManager:
    """Manages forward hooks for a module and its submodules with automatic cleanup."""
    
    def __init__(self):
        """Initialize the HooksManager.
        """
        self.hooks: Dict[nn.Module, Any] = {}

    def register_forward_hook(self, 
                              module: nn.Module,
                              hook_fn: Callable) -> None:
        """Register a forward hook on the specified module.
        
        Args:
            hook_fn: The hook function to register
        """
        # Remove existing hook if present
        if module in self.hooks:
            self.hooks[module].remove()
            
        handle = module.register_forward_hook(hook_fn)
        self.hooks[module] = handle
    
    def remove_hook(self, module: nn.Module) -> None:
        """Remove the hook from the specified module.
        
        Args:
            target: Same as register method
        """
        if module in self.hooks:
            self.hooks[module].remove()
            del self.hooks[module]
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def __len__(self) -> int:
        """Get count of registered hooks."""
        return len(self.hooks)
    
    def __del__(self):
        """Cleanup: remove all hooks when object is destroyed."""
        self.remove_all_hooks()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.remove_all_hooks()