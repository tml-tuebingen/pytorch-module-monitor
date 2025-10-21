"""
Basic ModuleMonitor tests with manual verification of computed statistics.

This test file focuses on verifying that the monitor correctly computes
activation, parameter, and gradient statistics by manually computing
the same statistics and comparing the results.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_module_monitor.monitor import ModuleMonitor
import logging


class TestMonitorBasicSetup:
    """Test basic monitor setup and initialization."""

    def test_monitor_init(self):
        """Test basic initialization of ModuleMonitor."""
        monitor = ModuleMonitor()
        assert monitor.module is None
        assert monitor.reference_module is None
        assert monitor.current_step is None
        assert not monitor.is_monitoring()

    def test_set_module(self, simple_linear_model):
        """Test setting a module to monitor."""
        monitor = ModuleMonitor()
        monitor.set_module(simple_linear_model)

        assert monitor.module is simple_linear_model
        # Verify hooks are registered on all submodules
        # The model has: fc1, relu, fc2 + root module = 4 modules total
        assert len(monitor.module_names) == 4

    def test_set_module_twice_fails(self, simple_linear_model):
        """Test that setting a module twice raises an error."""
        monitor = ModuleMonitor()
        monitor.set_module(simple_linear_model)

        with pytest.raises(ValueError, match="Module has already been set"):
            monitor.set_module(simple_linear_model)

    def test_begin_step(self, simple_linear_model):
        """Test begin_step and step monitoring."""
        monitor = ModuleMonitor(monitor_step_fn=lambda step: step % 2 == 0)
        monitor.set_module(simple_linear_model)

        # Step 0 should be monitored (0 % 2 == 0)
        is_monitored = monitor.begin_step(0)
        assert is_monitored
        assert monitor.is_monitoring()
        assert monitor.current_step == 0

        # End step
        monitor.end_step()
        assert not monitor.is_monitoring()

        # Step 1 should not be monitored (1 % 2 != 0)
        is_monitored = monitor.begin_step(1)
        assert not is_monitored
        assert not monitor.is_monitoring()


class TestActivationStatistics:
    """Test that activation statistics are computed correctly."""

    def test_activation_mean_and_std_manual_verification(self, device):
        """
        Manually compute activation statistics and verify the monitor
        computes the same values.
        """
        # Create a simple model with known structure
        class TwoLayerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = TwoLayerNet().to(device)

        # Set known weights for reproducibility
        torch.manual_seed(42)
        with torch.no_grad():
            model.fc1.weight.data = torch.randn_like(model.fc1.weight.data)
            model.fc2.weight.data = torch.randn_like(model.fc2.weight.data)

        # Create known input
        torch.manual_seed(123)
        input_data = torch.randn(3, 4, device=device)  # batch_size=3, input_dim=4

        # Set up monitor with mean and std metrics
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model)
        monitor.add_activation_metric("mean", lambda x: x.mean())
        monitor.add_activation_metric("std", lambda x: x.std())
        monitor.add_activation_metric("abs_max", lambda x: x.abs().max())

        # Manually compute activations
        with torch.no_grad():
            # First layer: fc1
            fc1_output = F.linear(input_data, model.fc1.weight, None)
            fc1_mean = fc1_output.mean().item()
            fc1_std = fc1_output.std().item()
            fc1_abs_max = fc1_output.abs().max().item()

            # ReLU
            relu_output = F.relu(fc1_output)
            relu_mean = relu_output.mean().item()
            relu_std = relu_output.std().item()
            relu_abs_max = relu_output.abs().max().item()

            # Second layer: fc2
            fc2_output = F.linear(relu_output, model.fc2.weight, None)
            fc2_mean = fc2_output.mean().item()
            fc2_std = fc2_output.std().item()
            fc2_abs_max = fc2_output.abs().max().item()

        # Run monitor
        monitor.begin_step(0)
        with torch.no_grad():
            output = model(input_data)
        monitor.end_step()

        # Get metrics from monitor
        metrics = monitor.get_step_metrics()

        # Verify fc1 activations
        assert "activation/fc1/mean" in metrics
        assert "activation/fc1/std" in metrics
        assert "activation/fc1/abs_max" in metrics
        assert abs(metrics["activation/fc1/mean"] - fc1_mean) < 1e-6
        assert abs(metrics["activation/fc1/std"] - fc1_std) < 1e-6
        assert abs(metrics["activation/fc1/abs_max"] - fc1_abs_max) < 1e-6

        # Verify relu activations
        assert "activation/relu/mean" in metrics
        assert "activation/relu/std" in metrics
        assert "activation/relu/abs_max" in metrics
        assert abs(metrics["activation/relu/mean"] - relu_mean) < 1e-6
        assert abs(metrics["activation/relu/std"] - relu_std) < 1e-6
        assert abs(metrics["activation/relu/abs_max"] - relu_abs_max) < 1e-6

        # Verify fc2 activations
        assert "activation/fc2/mean" in metrics
        assert "activation/fc2/std" in metrics
        assert "activation/fc2/abs_max" in metrics
        assert abs(metrics["activation/fc2/mean"] - fc2_mean) < 1e-6
        assert abs(metrics["activation/fc2/std"] - fc2_std) < 1e-6
        assert abs(metrics["activation/fc2/abs_max"] - fc2_abs_max) < 1e-6

    def test_activation_metric_with_regex_filter(self, device):
        """Test that activation metrics respect regex filters."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.fc2 = nn.Linear(8, 2, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = SimpleNet().to(device)
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model)

        # Only monitor fc1 layer
        monitor.add_activation_metric("mean", lambda x: x.mean(), metric_regex=r"fc1")

        input_data = torch.randn(2, 4, device=device)
        monitor.begin_step(0)
        with torch.no_grad():
            output = model(input_data)
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # Should have fc1 but not fc2
        assert "activation/fc1/mean" in metrics
        assert "activation/fc2/mean" not in metrics


class TestParameterStatistics:
    """Test that parameter statistics are computed correctly."""

    def test_parameter_mean_norm_manual_verification(self, device):
        """
        Manually compute parameter statistics and verify the monitor
        computes the same values.
        """
        class TwoLayerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.fc2 = nn.Linear(8, 2, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = TwoLayerNet().to(device)

        # Set known weights
        torch.manual_seed(42)
        with torch.no_grad():
            model.fc1.weight.data = torch.randn_like(model.fc1.weight.data)
            model.fc2.weight.data = torch.randn_like(model.fc2.weight.data)

        # Manually compute parameter statistics
        fc1_weight_mean = model.fc1.weight.data.mean().item()
        fc1_weight_std = model.fc1.weight.data.std().item()
        fc1_weight_norm = model.fc1.weight.data.norm().item()

        fc2_weight_mean = model.fc2.weight.data.mean().item()
        fc2_weight_std = model.fc2.weight.data.std().item()
        fc2_weight_norm = model.fc2.weight.data.norm().item()

        # Set up monitor
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model)
        monitor.add_parameter_metric("mean", lambda x: x.mean())
        monitor.add_parameter_metric("std", lambda x: x.std())
        monitor.add_parameter_metric("norm", lambda x: x.norm())

        # Run monitor
        input_data = torch.randn(2, 4, device=device)
        monitor.begin_step(0)
        with torch.no_grad():
            output = model(input_data)
        monitor.monitor_parameters()
        monitor.end_step()

        # Get metrics
        metrics = monitor.get_step_metrics()

        # Verify fc1 parameters
        assert "parameter/fc1.weight/mean" in metrics
        assert "parameter/fc1.weight/std" in metrics
        assert "parameter/fc1.weight/norm" in metrics
        assert abs(metrics["parameter/fc1.weight/mean"] - fc1_weight_mean) < 1e-6
        assert abs(metrics["parameter/fc1.weight/std"] - fc1_weight_std) < 1e-6
        assert abs(metrics["parameter/fc1.weight/norm"] - fc1_weight_norm) < 1e-6

        # Verify fc2 parameters
        assert "parameter/fc2.weight/mean" in metrics
        assert "parameter/fc2.weight/std" in metrics
        assert "parameter/fc2.weight/norm" in metrics
        assert abs(metrics["parameter/fc2.weight/mean"] - fc2_weight_mean) < 1e-6
        assert abs(metrics["parameter/fc2.weight/std"] - fc2_weight_std) < 1e-6
        assert abs(metrics["parameter/fc2.weight/norm"] - fc2_weight_norm) < 1e-6


class TestGradientStatistics:
    """Test that gradient statistics are computed correctly."""

    def test_gradient_mean_norm_manual_verification(self, device):
        """
        Manually compute gradient statistics and verify the monitor
        computes the same values.
        """
        class TwoLayerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.fc2 = nn.Linear(8, 2, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = TwoLayerNet().to(device)

        # Set known weights
        torch.manual_seed(42)
        with torch.no_grad():
            model.fc1.weight.data = torch.randn_like(model.fc1.weight.data)
            model.fc2.weight.data = torch.randn_like(model.fc2.weight.data)

        # Set up monitor
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model)
        monitor.add_gradient_metric("mean", lambda x: x.mean())
        monitor.add_gradient_metric("std", lambda x: x.std())
        monitor.add_gradient_metric("norm", lambda x: x.norm())

        # Run forward and backward pass
        torch.manual_seed(123)
        input_data = torch.randn(3, 4, device=device, requires_grad=False)
        target = torch.randn(3, 2, device=device)

        monitor.begin_step(0)
        output = model(input_data)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Manually compute gradient statistics BEFORE monitor reads them
        fc1_grad_mean = model.fc1.weight.grad.detach().mean().item()
        fc1_grad_std = model.fc1.weight.grad.detach().std().item()
        fc1_grad_norm = model.fc1.weight.grad.detach().norm().item()

        fc2_grad_mean = model.fc2.weight.grad.detach().mean().item()
        fc2_grad_std = model.fc2.weight.grad.detach().std().item()
        fc2_grad_norm = model.fc2.weight.grad.detach().norm().item()

        # Monitor gradients
        monitor.monitor_gradients()
        monitor.end_step()

        # Get metrics
        metrics = monitor.get_step_metrics()

        # Verify fc1 gradients
        assert "gradient/fc1.weight/mean" in metrics
        assert "gradient/fc1.weight/std" in metrics
        assert "gradient/fc1.weight/norm" in metrics
        assert abs(metrics["gradient/fc1.weight/mean"] - fc1_grad_mean) < 1e-6
        assert abs(metrics["gradient/fc1.weight/std"] - fc1_grad_std) < 1e-6
        assert abs(metrics["gradient/fc1.weight/norm"] - fc1_grad_norm) < 1e-6

        # Verify fc2 gradients
        assert "gradient/fc2.weight/mean" in metrics
        assert "gradient/fc2.weight/std" in metrics
        assert "gradient/fc2.weight/norm" in metrics
        assert abs(metrics["gradient/fc2.weight/mean"] - fc2_grad_mean) < 1e-6
        assert abs(metrics["gradient/fc2.weight/std"] - fc2_grad_std) < 1e-6
        assert abs(metrics["gradient/fc2.weight/norm"] - fc2_grad_norm) < 1e-6


class TestFullTrainingLoop:
    """Test a full training loop with all statistics."""

    def test_complete_training_step_with_manual_verification(self, device):
        """
        Test a complete training step with activation, parameter, and gradient
        monitoring, manually verifying all computed statistics.
        """
        # Create model
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5, bias=False)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(5, 2, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = SimpleNet().to(device)

        # Initialize weights
        torch.manual_seed(42)
        with torch.no_grad():
            model.fc1.weight.data.normal_(0, 0.1)
            model.fc2.weight.data.normal_(0, 0.1)

        # Set up monitor
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model)

        # Add metrics
        monitor.add_activation_metric("mean", lambda x: x.mean())
        monitor.add_parameter_metric("norm", lambda x: x.norm())
        monitor.add_gradient_metric("norm", lambda x: x.norm())

        # Create data
        torch.manual_seed(100)
        input_data = torch.randn(4, 3, device=device)
        target = torch.randn(4, 2, device=device)

        # ===== MANUAL COMPUTATION =====
        with torch.no_grad():
            # Activations
            fc1_activation = F.linear(input_data, model.fc1.weight, None)
            fc1_activation_mean = fc1_activation.mean().item()

            relu_activation = F.relu(fc1_activation)
            relu_activation_mean = relu_activation.mean().item()

            fc2_activation = F.linear(relu_activation, model.fc2.weight, None)
            fc2_activation_mean = fc2_activation.mean().item()

        # Parameters
        fc1_param_norm = model.fc1.weight.data.norm().item()
        fc2_param_norm = model.fc2.weight.data.norm().item()

        # Gradients (need to do forward/backward first)
        output = model(input_data)
        loss = F.mse_loss(output, target)
        loss.backward()

        fc1_grad_norm = model.fc1.weight.grad.detach().norm().item()
        fc2_grad_norm = model.fc2.weight.grad.detach().norm().item()

        # Zero gradients to start fresh for monitor
        model.zero_grad()

        # ===== MONITOR COMPUTATION =====
        monitor.begin_step(0)
        output = model(input_data)
        loss = F.mse_loss(output, target)
        loss.backward()
        monitor.monitor_parameters()
        monitor.monitor_gradients()
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        # Activations
        assert abs(metrics["activation/fc1/mean"] - fc1_activation_mean) < 1e-5
        assert abs(metrics["activation/relu/mean"] - relu_activation_mean) < 1e-5
        assert abs(metrics["activation/fc2/mean"] - fc2_activation_mean) < 1e-5

        # Parameters
        assert abs(metrics["parameter/fc1.weight/norm"] - fc1_param_norm) < 1e-5
        assert abs(metrics["parameter/fc2.weight/norm"] - fc2_param_norm) < 1e-5

        # Gradients
        assert abs(metrics["gradient/fc1.weight/norm"] - fc1_grad_norm) < 1e-5
        assert abs(metrics["gradient/fc2.weight/norm"] - fc2_grad_norm) < 1e-5
