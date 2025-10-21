# torch-module-monitor

[![PyPI version](https://badge.fury.io/py/torch-module-monitor.svg)](https://badge.fury.io/py/torch-module-monitor)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep-dive diagnostics for PyTorch model training. Monitor activations, parameters, and gradients across all layers during training with minimal code changes.

## Features

- 🔍 **Comprehensive Monitoring**: Track activations, parameters, and gradients for any PyTorch model
- 📊 **Flexible Metrics**: Define custom metrics with regex-based filtering for specific layers
- 🔄 **Reference Module Comparison**: Compare your trained model against a reference (e.g., model at initialization)
- 🎯 **Refined Coordinate Check**: Decompose activation changes into weight updates vs. input drift ([Yang et al., 2025](https://arxiv.org/abs/2505.22491))
- 👁️ **Attention Monitoring**: Built-in support for monitoring multi-head attention mechanisms
- 💾 **Storage & Export**: Aggregate metrics across micro-batches and export to HDF5
- ⚡ **Step-Based Monitoring**: Selectively monitor specific training steps to reduce overhead
- 🔌 **Framework Agnostic**: Works with Weights & Biases, TensorBoard, or any logging framework

## Installation

```bash
pip install torch-module-monitor
```

### Optional Dependencies

```bash
# For Weights & Biases integration
pip install torch-module-monitor[wandb]

# For TensorBoard integration
pip install torch-module-monitor[tb]

# For development
pip install torch-module-monitor[dev]
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch_module_monitor import ModuleMonitor

# Create your model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)

# Set up the monitor
monitor = ModuleMonitor(
    monitor_step_fn=lambda step: step % 10 == 0  # Monitor every 10 steps
)
monitor.set_module(model)

# Add metrics
monitor.add_activation_metric("mean", lambda x: x.mean())
monitor.add_activation_metric("std", lambda x: x.std())
monitor.add_parameter_metric("norm", lambda x: x.norm())
monitor.add_gradient_metric("norm", lambda x: x.norm())

# Training loop
optimizer = torch.optim.Adam(model.parameters())

for step, (inputs, targets) in enumerate(dataloader):
    # Begin monitoring step
    monitor.begin_step(step)

    # Forward pass (activations are automatically captured)
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Monitor parameters and gradients
    monitor.monitor_parameters()
    monitor.monitor_gradients()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Finalize step and get metrics
    monitor.end_step()

    if monitor.is_step_monitored(step):
        metrics = monitor.get_step_metrics()
        # Log to your preferred backend (wandb, tensorboard, etc.)
        wandb.log(metrics)
```

## Core Concepts

### Activation Monitoring

Activations (layer outputs) are automatically captured during forward passes via PyTorch hooks. Metrics are logged as `activation/{module_name}/{metric_name}`.

```python
# Monitor L2 norm of all layer activations
monitor.add_activation_metric("l2_norm", lambda x: x.norm(dim=-1).mean())

# Monitor only attention layers
monitor.add_activation_metric(
    "attention_mean",
    lambda x: x.mean(),
    metric_regex=r".*attention.*"  # Regex to match layer names
)
```

### Reference Module Comparison

Compare your model against a reference (e.g., initialization) to track parameter/activation drift:

```python
import copy

# Save reference at initialization
reference_model = copy.deepcopy(model)
monitor.set_reference_module(reference_model)

# Add difference metrics
monitor.add_activation_difference_metric(
    "l2_distance",
    lambda act, ref_act: (act - ref_act).norm(dim=-1).mean()
)

monitor.add_parameter_difference_metric(
    "l2_distance",
    lambda param, ref_param: (param - ref_param).norm()
)

# During training, run reference model first
with monitor.no_monitor():
    _ = reference_model(inputs)  # Not monitored, just stores activations
outputs = model(inputs)  # Monitored + compares with reference
```

### Refined Coordinate Check (RCC)

Decompose activation changes into weight updates vs. input drift for muP coordinate checking:

```python
from torch_module_monitor import RefinedCoordinateCheck

# Set up RCC (requires reference module)
rcc = RefinedCoordinateCheck(monitor)

# In your training loop
monitor.begin_step(step)

# Run both models
with monitor.no_monitor():
    _ = reference_model(inputs)
outputs = model(inputs)

# Perform RCC analysis
rcc.refined_coordinate_check()  # Logs (W_t-W_0)x_t and W_0(x_t-x_0)

monitor.end_step()
```

See the [RCC paper](https://arxiv.org/abs/2505.22491) for details on the methodology.

### Micro-batch Support

For gradient accumulation, use `after_micro_batch()`:

```python
monitor.begin_step(step)

for micro_batch in batches:
    outputs = model(micro_batch)
    loss = criterion(outputs, targets)
    loss.backward()

    monitor.after_micro_batch()  # Clear reference activations

monitor.monitor_gradients()
optimizer.step()
monitor.end_step()  # Aggregates metrics across all micro-batches
```

## Advanced Usage

### Attention Monitoring

```python
from torch_module_monitor import monitor_scaled_dot_product_attention

class MyAttention(nn.Module, MonitorMixin):
    def forward(self, x):
        q, k, v = self.get_qkv(x)

        # Standard attention
        output = F.scaled_dot_product_attention(q, k, v)

        # Monitor attention internals
        if self.is_monitoring:
            monitor_scaled_dot_product_attention(
                self.get_module_monitor(),
                module=self,
                query=q, key=k, value=v,
                activation=output
            )

        return output
```

### Custom Metrics in Modules

```python
from torch_module_monitor import MonitorMixin

class CustomLayer(nn.Module, MonitorMixin):
    def forward(self, x):
        output = self.transform(x)

        if self.is_monitoring:
            # Log custom intermediate values
            self.get_module_monitor().log_tensor(
                "custom_metric/my_layer/intermediate",
                some_intermediate_value
            )

        return output
```

### Export to HDF5

```python
# Get all metrics
all_metrics = monitor.get_all_metrics()

# Export via storage manager
monitor.storage.save_hdf5("training_metrics.h5")

# Load later for analysis
from torch_module_monitor import StorageManager
metrics = StorageManager.load_hdf5("training_metrics.h5")
```

## Documentation

Full documentation is available at [torch-module-monitor.readthedocs.io](https://torch-module-monitor.readthedocs.io).

## Citation

If you use the Refined Coordinate Check in your research, please cite:

```bibtex
@article{yang2025refined,
  title={Refined Coordinate Checks for Feature Learning},
  author={Yang, Greg and Malladi, Sadhika and Bordt, Sebastian and Huh, Minyoung and Nanda, Neel and Gao, Brando and Hu, Edward and Timbers, Fern and Gur-Ari, Guy and Sholto, Douglas and others},
  journal={arXiv preprint arXiv:2505.22491},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Refined Coordinate Check implementation based on [Yang et al., 2025](https://arxiv.org/abs/2505.22491)
- Developed at the [Tübingen Machine Learning Lab](https://tml.cs.uni-tuebingen.de/)
