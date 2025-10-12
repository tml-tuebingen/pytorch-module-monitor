import pytest
import torch
import torch.nn as nn
import re
from unittest.mock import Mock, call
from typing import List, Tuple

# add ../pytorch_module_monitor to path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/pytorch_module_monitor")))
from hooks import ModuleHooksManager


class SimpleModel(nn.Module):
    """Simple test model with various layer types."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention = nn.MultiheadAttention(64, 8)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        return x


class TestModuleHooksManager:
    """Test suite for ModuleHooksManager."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    @pytest.fixture
    def mock_hook(self):
        """Create a mock hook function."""
        return Mock()
    
    def test_initialization(self, simple_model):
        """Test basic initialization."""
        manager = ModuleHooksManager(simple_model)
        
        # Debug: print all module names to verify count
        # for module, name in manager.module_names.items():
        #     print(f"{name}: {module.__class__.__name__}")
        
        # Check that module names are registered (7 submodules + 1 root = 8)
        assert len(manager.module_names) == 8
        assert manager.module == simple_model
        assert len(manager.hooks) == 0
        
        # Check root module naming
        assert manager.module_names[simple_model] == "[root module]"
    
    def test_custom_name_formatter(self, simple_model):
        """Test custom name formatting function."""
        def custom_formatter(name: str) -> str:
            if name == "":
                return "ROOT"
            return f"custom_{name}"
        
        manager = ModuleHooksManager(simple_model, format_module_name_fn=custom_formatter)
        
        assert manager.module_names[simple_model] == "ROOT"
        assert manager.module_names[simple_model.conv1] == "custom_conv1"
        assert manager.module_names[simple_model.linear1] == "custom_linear1"
    
    def test_register_hook_all_modules(self, simple_model, mock_hook):
        """Test registering a hook on all modules."""
        manager = ModuleHooksManager(simple_model)
        manager.register_forward_hook(mock_hook)
        
        # Should register on all 8 modules
        assert len(manager.hooks) == 8
        
        # Trigger forward pass to check hooks are called
        x = torch.randn(1, 3, 32, 32)
        simple_model.conv1(x)
        
        # Hook should have been called
        assert mock_hook.called
    
    def test_register_hook_string_target(self, simple_model, mock_hook):
        """Test registering hook with string target (substring match)."""
        manager = ModuleHooksManager(simple_model)
        manager.register_forward_hook(mock_hook, target="linear")
        
        # Should only register on linear1 and linear2
        assert len(manager.hooks) == 2
        assert simple_model.linear1 in manager.hooks
        assert simple_model.linear2 in manager.hooks
        assert simple_model.conv1 not in manager.hooks
    
    def test_register_hook_regex_target(self, simple_model, mock_hook):
        """Test registering hook with regex pattern."""
        manager = ModuleHooksManager(simple_model)
        pattern = re.compile(r"linear\d+")
        manager.register_forward_hook(mock_hook, target=pattern)
        
        # Should match linear1 and linear2
        assert len(manager.hooks) == 2
        assert simple_model.linear1 in manager.hooks
        assert simple_model.linear2 in manager.hooks
    
    def test_remove_hooks_by_target(self, simple_model, mock_hook):
        """Test removing hooks by target."""
        manager = ModuleHooksManager(simple_model)
        
        # Register on all modules
        manager.register_forward_hook(mock_hook)
        assert len(manager.hooks) == 8
        
        # Remove only linear modules
        manager.remove(target="linear")
        assert len(manager.hooks) == 6
        assert simple_model.linear1 not in manager.hooks
        assert simple_model.linear2 not in manager.hooks
        assert simple_model.conv1 in manager.hooks
    
    def test_remove_all_hooks(self, simple_model, mock_hook):
        """Test removing all hooks."""
        manager = ModuleHooksManager(simple_model)
        manager.register_forward_hook(mock_hook)
        
        assert len(manager.hooks) == 8
        manager.remove_all()
        assert len(manager.hooks) == 0
    
    def test_hook_replacement(self, simple_model):
        """Test that registering a new hook replaces the old one."""
        manager = ModuleHooksManager(simple_model)
        
        hook1 = Mock()
        hook2 = Mock()
        
        # Register first hook
        manager.register_forward_hook(hook1, target="conv1")
        assert len(manager.hooks) == 1
        
        # Register second hook on same target
        manager.register_forward_hook(hook2, target="conv1")
        assert len(manager.hooks) == 1  # Still only one
        
        # Trigger forward and check only hook2 is called
        x = torch.randn(1, 3, 32, 32)
        simple_model.conv1(x)
        
        assert not hook1.called
        assert hook2.called
    
    def test_get_module_name(self, simple_model):
        """Test getting module names."""
        manager = ModuleHooksManager(simple_model)
        
        assert manager.get_module_name(simple_model) == "[root module]"
        assert manager.get_module_name(simple_model.conv1) == "conv1"
        assert manager.get_module_name(simple_model.attention) == "attention"
        
        # Test with unknown module
        unknown_module = nn.Linear(10, 10)
        assert manager.get_module_name(unknown_module) is None
    
    def test_len_method(self, simple_model, mock_hook):
        """Test __len__ method."""
        manager = ModuleHooksManager(simple_model)
        
        assert len(manager) == 0
        
        manager.register_forward_hook(mock_hook, target="linear")
        assert len(manager) == 2
        
        manager.register_forward_hook(mock_hook, target="conv")
        assert len(manager) == 3  # 2 linear + 1 conv
    
    def test_context_manager(self, simple_model, mock_hook):
        """Test context manager functionality."""
        with ModuleHooksManager(simple_model) as manager:
            manager.register_forward_hook(mock_hook)
            assert len(manager.hooks) == 8
        
        # Hooks should be removed after exiting context
        assert len(manager.hooks) == 0
    
    def test_automatic_cleanup_on_delete(self, simple_model, mock_hook):
        """Test automatic cleanup when object is deleted."""
        manager = ModuleHooksManager(simple_model)
        manager.register_forward_hook(mock_hook)
        
        # Store hook handles to check they're removed
        hook_handles = list(manager.hooks.values())
        
        # Delete manager
        del manager
        
        # This is hard to test directly, but we can check that 
        # registering new hooks still works (no interference)
        new_manager = ModuleHooksManager(simple_model)
        new_hook = Mock()
        new_manager.register_forward_hook(new_hook)
        
        x = torch.randn(1, 3, 32, 32)
        simple_model.conv1(x)
        
        # Only new hook should be called
        assert new_hook.called
        assert not mock_hook.called
    
    def test_invalid_target_type(self, simple_model):
        """Test that invalid target types raise ValueError."""
        manager = ModuleHooksManager(simple_model)
        
        with pytest.raises(ValueError, match="Invalid target type"):
            manager.register_forward_hook(Mock(), target=123)
        
        with pytest.raises(ValueError, match="Invalid target type"):
            manager.register_forward_hook(Mock(), target=[])
    
    def test_hook_execution_order(self, simple_model):
        """Test that hooks receive correct arguments."""
        manager = ModuleHooksManager(simple_model)
        
        captured_args = []
        
        def capture_hook(module, input, output):
            captured_args.append({
                'module': module,
                'input_shape': input[0].shape if input else None,
                'output_shape': output.shape if hasattr(output, 'shape') else None
            })
            
        manager.register_forward_hook(capture_hook, target="conv1")
        
        x = torch.randn(1, 3, 32, 32)
        output = simple_model.conv1(x)
        
        assert len(captured_args) == 1
        assert captured_args[0]['module'] == simple_model.conv1
        assert captured_args[0]['input_shape'] == x.shape
        assert captured_args[0]['output_shape'] == output.shape


# Integration test example that would go in your ModuleMonitor tests
class TestModuleHooksIntegration:
    """Integration tests showing how ModuleHooksManager works with ModuleMonitor."""
    
    def test_with_module_monitor_workflow(self):
        """Test integration with ModuleMonitor-like workflow."""
        model = SimpleModel()
        
        # Simulate what ModuleMonitor does
        activations = {}
        
        def store_activations(module, input, output):
            name = hooks.get_module_name(module)
            if name:
                activations[name] = output.detach()
        
        hooks = ModuleHooksManager(model)
        hooks.register_forward_hook(store_activations)
        
        # Run forward pass
        x = torch.randn(1, 3, 32, 32)
        _ = model.conv1(x)
        
        # Check activations were stored
        assert "conv1" in activations
        assert activations["conv1"].shape[1] == 64  # Conv output channels
        
        # Clean up
        hooks.remove_all()
        activations.clear()
        
        # Verify hooks are removed
        _ = model.conv1(x)
        assert len(activations) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])