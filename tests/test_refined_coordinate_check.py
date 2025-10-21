"""
Test the Refined Coordinate Check (RCC) with manual verification.

The RCC decomposes activation changes into:
- (W_t - W_0) @ x_t: Change due to weight updates
- W_0 @ (x_t - x_0): Change due to input changes

This test manually computes these decompositions and verifies the monitor
logs the correct L2 norms.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_module_monitor.monitor import ModuleMonitor
from pytorch_module_monitor.refined_coordinate_check import RefinedCoordinateCheck, l2_norm
import copy


class TestRCCBasicSetup:
    """Test basic RCC setup and initialization."""

    def test_rcc_requires_module(self):
        """Test that RCC requires a module to be set."""
        monitor = ModuleMonitor()

        with pytest.raises(ValueError, match="Set the module to monitor"):
            RefinedCoordinateCheck(monitor)

    def test_rcc_requires_reference_module(self, simple_linear_model):
        """Test that RCC requires a reference module."""
        monitor = ModuleMonitor()
        monitor.set_module(simple_linear_model)

        with pytest.raises(ValueError, match="Set the reference module"):
            RefinedCoordinateCheck(monitor)

    def test_rcc_init_success(self, simple_linear_model):
        """Test successful RCC initialization."""
        monitor = ModuleMonitor()
        monitor.set_module(simple_linear_model)

        # Create reference module (copy of initial state)
        reference_model = copy.deepcopy(simple_linear_model)
        monitor.set_reference_module(reference_model)

        # Should not raise
        rcc = RefinedCoordinateCheck(monitor)
        assert rcc.module is simple_linear_model
        assert rcc.reference_module is reference_model


class TestRCCLinearLayer:
    """Test RCC on linear layers with manual verification."""

    def test_rcc_linear_layer_manual_verification(self, device):
        """
        Manually compute RCC decomposition for a linear layer and verify
        the monitor computes the same values.

        RCC computes:
        - (W_t - W_0) @ x_t = W_t @ x_t - W_0 @ x_t
        - W_0 @ (x_t - x_0) = W_0 @ x_t - W_0 @ x_0

        And logs L2 norms of these terms.
        """
        # Create a simple linear model
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 8, bias=False)

            def forward(self, x):
                return self.fc(x)

        # Initialize model at t=0
        torch.manual_seed(42)
        model_t0 = SimpleLinear().to(device)
        W_0 = model_t0.fc.weight.data.clone()

        # Create model at t=1 (after some training)
        model_t1 = copy.deepcopy(model_t0)
        # Simulate weight update
        with torch.no_grad():
            model_t1.fc.weight.data += torch.randn_like(model_t1.fc.weight.data) * 0.01
        W_t = model_t1.fc.weight.data.clone()

        # Create inputs
        torch.manual_seed(100)
        x_0 = torch.randn(3, 4, device=device)  # batch_size=3, input_dim=4

        torch.manual_seed(200)
        x_t = torch.randn(3, 4, device=device)

        # ===== MANUAL COMPUTATION =====
        with torch.no_grad():
            # Reference forward: W_0 @ x_0
            W0_x0 = F.linear(x_0, W_0, None)

            # Monitored forward: W_t @ x_t
            Wt_xt = F.linear(x_t, W_t, None)

            # Additional forward for RCC: W_0 @ x_t
            W0_xt = F.linear(x_t, W_0, None)

            # Decomposition:
            # (W_t - W_0) @ x_t = W_t @ x_t - W_0 @ x_t
            weight_change_term = Wt_xt - W0_xt
            weight_change_l2norm = l2_norm(weight_change_term)

            # W_0 @ (x_t - x_0) = W_0 @ x_t - W_0 @ x_0
            input_change_term = W0_xt - W0_x0
            input_change_l2norm = l2_norm(input_change_term)

            # Input norms
            xt_l2norm = l2_norm(x_t)
            xt_minus_x0_l2norm = l2_norm(x_t - x_0)

        # ===== MONITOR COMPUTATION =====
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model_t1)
        monitor.set_reference_module(model_t0)

        rcc = RefinedCoordinateCheck(monitor)

        monitor.begin_step(0)

        # Reference forward pass
        with torch.no_grad():
            ref_out = model_t0(x_0)

        # Monitored forward pass
        with torch.no_grad():
            mon_out = model_t1(x_t)

        # Perform RCC
        rcc.refined_coordinate_check()

        monitor.end_step()

        # Get metrics
        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        # The metrics should be tensors (one value per batch element)
        # We need to compare them element-wise

        # RCC metrics are aggregated to scalars (mean across batch)
        # Convert our manually computed tensors to scalars for comparison
        assert "RCC (W_t-W_0)x_t/fc/l2norm" in metrics
        rcc_weight_change = metrics["RCC (W_t-W_0)x_t/fc/l2norm"]
        assert abs(rcc_weight_change - weight_change_l2norm.mean().item()) < 1e-5

        assert "RCC W_0(x_t-x_0)/fc/l2norm" in metrics
        rcc_input_change = metrics["RCC W_0(x_t-x_0)/fc/l2norm"]
        assert abs(rcc_input_change - input_change_l2norm.mean().item()) < 1e-5

        assert "RCC x_t/fc/l2norm" in metrics
        rcc_xt = metrics["RCC x_t/fc/l2norm"]
        assert abs(rcc_xt - xt_l2norm.mean().item()) < 1e-5

        assert "RCC (x_t-x_0)/fc/l2norm" in metrics
        rcc_xt_minus_x0 = metrics["RCC (x_t-x_0)/fc/l2norm"]
        assert abs(rcc_xt_minus_x0 - xt_minus_x0_l2norm.mean().item()) < 1e-5

    def test_rcc_linear_layer_with_bias_manual_verification(self, device):
        """
        Test RCC on linear layer with bias, verifying both regular and
        bias-free (weight-only) metrics.
        """
        # Create a simple linear model with bias
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 8, bias=True)

            def forward(self, x):
                return self.fc(x)

        # Initialize model at t=0
        torch.manual_seed(42)
        model_t0 = SimpleLinear().to(device)
        W_0 = model_t0.fc.weight.data.clone()
        b_0 = model_t0.fc.bias.data.clone()

        # Create model at t=1
        model_t1 = copy.deepcopy(model_t0)
        with torch.no_grad():
            model_t1.fc.weight.data += torch.randn_like(model_t1.fc.weight.data) * 0.01
            model_t1.fc.bias.data += torch.randn_like(model_t1.fc.bias.data) * 0.01
        W_t = model_t1.fc.weight.data.clone()
        b_t = model_t1.fc.bias.data.clone()

        # Create inputs
        torch.manual_seed(100)
        x_0 = torch.randn(3, 4, device=device)

        torch.manual_seed(200)
        x_t = torch.randn(3, 4, device=device)

        # ===== MANUAL COMPUTATION =====
        with torch.no_grad():
            # With bias
            W0_x0 = F.linear(x_0, W_0, b_0)
            Wt_xt = F.linear(x_t, W_t, b_t)
            W0_xt = F.linear(x_t, W_0, b_0)

            weight_change_l2norm = l2_norm(Wt_xt - W0_xt)
            input_change_l2norm = l2_norm(W0_xt - W0_x0)

            # Without bias (weight only)
            W0_x0_nobias = W0_x0 - b_0
            Wt_xt_nobias = Wt_xt - b_t
            W0_xt_nobias = W0_xt - b_0

            weight_change_nobias_l2norm = l2_norm(Wt_xt_nobias - W0_xt_nobias)
            input_change_nobias_l2norm = l2_norm(W0_xt_nobias - W0_x0_nobias)

            # Bias-free output norms
            Wt_xt_nobias_l2norm = l2_norm(Wt_xt_nobias)
            W0_x0_nobias_l2norm = l2_norm(W0_x0_nobias)

        # ===== MONITOR COMPUTATION =====
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model_t1)
        monitor.set_reference_module(model_t0)

        rcc = RefinedCoordinateCheck(monitor)

        monitor.begin_step(0)
        with torch.no_grad():
            ref_out = model_t0(x_0)
            mon_out = model_t1(x_t)
        rcc.refined_coordinate_check()
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        # With bias
        assert abs(
            metrics["RCC (W_t-W_0)x_t/fc/l2norm"] -
            weight_change_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/fc/l2norm"] -
            input_change_l2norm.mean().item()
        ) < 1e-5

        # Without bias (weight only)
        assert "RCC (W_t-W_0)x_t/fc.weight/l2norm" in metrics
        assert abs(
            metrics["RCC (W_t-W_0)x_t/fc.weight/l2norm"] -
            weight_change_nobias_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/fc.weight/l2norm"] -
            input_change_nobias_l2norm.mean().item()
        ) < 1e-5

        # Bias-free output norms
        assert abs(
            metrics["RCC (W_t-W_0)x_t/fc.weight/W_t x_t/l2norm"] -
            Wt_xt_nobias_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/fc.weight/W_0 x_0/l2norm"] -
            W0_x0_nobias_l2norm.mean().item()
        ) < 1e-5


class TestRCCMultiLayerNetwork:
    """Test RCC on multi-layer networks."""

    def test_rcc_two_layer_network_manual_verification(self, device):
        """
        Test RCC on a two-layer network, verifying decomposition for each layer.
        """
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

        # Initialize models
        torch.manual_seed(42)
        model_t0 = TwoLayerNet().to(device)

        model_t1 = copy.deepcopy(model_t0)
        # Simulate training
        with torch.no_grad():
            model_t1.fc1.weight.data += torch.randn_like(model_t1.fc1.weight.data) * 0.01
            model_t1.fc2.weight.data += torch.randn_like(model_t1.fc2.weight.data) * 0.01

        # Create inputs
        torch.manual_seed(100)
        x_0 = torch.randn(2, 4, device=device)

        torch.manual_seed(200)
        x_t = torch.randn(2, 4, device=device)

        # ===== MANUAL COMPUTATION FOR FC1 =====
        with torch.no_grad():
            # Reference forward
            fc1_W0_x0 = F.linear(x_0, model_t0.fc1.weight, None)
            relu_W0_x0 = F.relu(fc1_W0_x0)

            # Monitored forward
            fc1_Wt_xt = F.linear(x_t, model_t1.fc1.weight, None)
            relu_Wt_xt = F.relu(fc1_Wt_xt)

            # RCC forward for fc1
            fc1_W0_xt = F.linear(x_t, model_t0.fc1.weight, None)

            # FC1 decomposition
            fc1_weight_change = l2_norm(fc1_Wt_xt - fc1_W0_xt)
            fc1_input_change = l2_norm(fc1_W0_xt - fc1_W0_x0)

            # ===== MANUAL COMPUTATION FOR FC2 =====
            # For fc2, inputs are the relu outputs
            fc2_W0_x0 = F.linear(relu_W0_x0, model_t0.fc2.weight, None)
            fc2_Wt_xt = F.linear(relu_Wt_xt, model_t1.fc2.weight, None)
            fc2_W0_xt = F.linear(relu_Wt_xt, model_t0.fc2.weight, None)

            # FC2 decomposition
            fc2_weight_change = l2_norm(fc2_Wt_xt - fc2_W0_xt)
            fc2_input_change = l2_norm(fc2_W0_xt - fc2_W0_x0)

        # ===== MONITOR COMPUTATION =====
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model_t1)
        monitor.set_reference_module(model_t0)

        rcc = RefinedCoordinateCheck(monitor)

        monitor.begin_step(0)
        with torch.no_grad():
            ref_out = model_t0(x_0)
            mon_out = model_t1(x_t)
        rcc.refined_coordinate_check()
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        # FC1 layer
        assert abs(
            metrics["RCC (W_t-W_0)x_t/fc1/l2norm"] -
            fc1_weight_change.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/fc1/l2norm"] -
            fc1_input_change.mean().item()
        ) < 1e-5

        # FC2 layer
        assert abs(
            metrics["RCC (W_t-W_0)x_t/fc2/l2norm"] -
            fc2_weight_change.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/fc2/l2norm"] -
            fc2_input_change.mean().item()
        ) < 1e-5


class TestRCCLayerNorm:
    """Test RCC on LayerNorm layers."""

    def test_rcc_layer_norm_manual_verification(self, device):
        """
        Test RCC on LayerNorm layer, verifying both regular and
        normalized input metrics.
        """
        class ModelWithLayerNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(4)

            def forward(self, x):
                return self.ln(x)

        # Initialize models
        torch.manual_seed(42)
        model_t0 = ModelWithLayerNorm().to(device)

        model_t1 = copy.deepcopy(model_t0)
        with torch.no_grad():
            model_t1.ln.weight.data += torch.randn_like(model_t1.ln.weight.data) * 0.01
            model_t1.ln.bias.data += torch.randn_like(model_t1.ln.bias.data) * 0.01

        # Create inputs
        torch.manual_seed(100)
        x_0 = torch.randn(2, 4, device=device)

        torch.manual_seed(200)
        x_t = torch.randn(2, 4, device=device)

        # ===== MANUAL COMPUTATION =====
        with torch.no_grad():
            # Full LayerNorm outputs
            W0_x0 = model_t0.ln(x_0)
            Wt_xt = model_t1.ln(x_t)
            W0_xt = model_t0.ln(x_t)

            # With bias
            weight_change_l2norm = l2_norm(Wt_xt - W0_xt)
            input_change_l2norm = l2_norm(W0_xt - W0_x0)

            # Without bias (weight only)
            W0_x0_nobias = W0_x0 - model_t0.ln.bias
            Wt_xt_nobias = Wt_xt - model_t1.ln.bias
            W0_xt_nobias = W0_xt - model_t0.ln.bias

            weight_change_nobias_l2norm = l2_norm(Wt_xt_nobias - W0_xt_nobias)
            input_change_nobias_l2norm = l2_norm(W0_xt_nobias - W0_x0_nobias)

            # Normalized inputs (before weight/bias application)
            xt_normalized = F.layer_norm(x_t, model_t1.ln.normalized_shape, None, None, model_t1.ln.eps)
            x0_normalized = F.layer_norm(x_0, model_t0.ln.normalized_shape, None, None, model_t0.ln.eps)

            xt_normalized_l2norm = l2_norm(xt_normalized)
            xt_minus_x0_normalized_l2norm = l2_norm(xt_normalized - x0_normalized)

        # ===== MONITOR COMPUTATION =====
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model_t1)
        monitor.set_reference_module(model_t0)

        rcc = RefinedCoordinateCheck(monitor)

        monitor.begin_step(0)
        with torch.no_grad():
            ref_out = model_t0(x_0)
            mon_out = model_t1(x_t)
        rcc.refined_coordinate_check()
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        # With bias
        assert abs(
            metrics["RCC (W_t-W_0)x_t/ln/l2norm"] -
            weight_change_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/ln/l2norm"] -
            input_change_l2norm.mean().item()
        ) < 1e-5

        # Without bias (weight only)
        assert abs(
            metrics["RCC (W_t-W_0)x_t/ln.weight/l2norm"] -
            weight_change_nobias_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/ln.weight/l2norm"] -
            input_change_nobias_l2norm.mean().item()
        ) < 1e-5

        # Normalized inputs
        assert abs(
            metrics["RCC (W_t-W_0)x_t/ln.weight/x_t/l2norm"] -
            xt_normalized_l2norm.mean().item()
        ) < 1e-5
        assert abs(
            metrics["RCC W_0(x_t-x_0)/ln.weight/x_t-x_0/l2norm"] -
            xt_minus_x0_normalized_l2norm.mean().item()
        ) < 1e-5


class TestRCCEmbedding:
    """Test RCC on Embedding layers."""

    def test_rcc_embedding_manual_verification(self, device):
        """
        Test RCC on Embedding layer. Note: Embedding only computes
        (W_t - W_0) @ x_t, not the input change term.
        """
        class ModelWithEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 8)

            def forward(self, x):
                return self.embed(x)

        # Initialize models
        torch.manual_seed(42)
        model_t0 = ModelWithEmbedding().to(device)

        model_t1 = copy.deepcopy(model_t0)
        with torch.no_grad():
            model_t1.embed.weight.data += torch.randn_like(model_t1.embed.weight.data) * 0.01

        # Create inputs (integer indices for embedding)
        torch.manual_seed(100)
        x_0 = torch.randint(0, 10, (2, 5), device=device)  # batch_size=2, seq_len=5

        torch.manual_seed(200)
        x_t = torch.randint(0, 10, (2, 5), device=device)

        # ===== MANUAL COMPUTATION =====
        with torch.no_grad():
            Wt_xt = model_t1.embed(x_t)
            W0_xt = model_t0.embed(x_t)

            # For embedding, only compute weight change term
            weight_change_l2norm = l2_norm(Wt_xt - W0_xt)

        # ===== MONITOR COMPUTATION =====
        monitor = ModuleMonitor(monitor_step_fn=lambda step: True)
        monitor.set_module(model_t1)
        monitor.set_reference_module(model_t0)

        rcc = RefinedCoordinateCheck(monitor)

        monitor.begin_step(0)
        with torch.no_grad():
            ref_out = model_t0(x_0)
            mon_out = model_t1(x_t)
        rcc.refined_coordinate_check()
        monitor.end_step()

        metrics = monitor.get_step_metrics()

        # ===== VERIFICATION =====
        assert "RCC (W_t-W_0)x_t/embed/l2norm" in metrics
        assert abs(
            metrics["RCC (W_t-W_0)x_t/embed/l2norm"] -
            weight_change_l2norm.mean().item()
        ) < 1e-5

        # Embedding should NOT have the input change term
        assert "RCC W_0(x_t-x_0)/embed/l2norm" not in metrics
