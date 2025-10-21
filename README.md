# torch-module-monitor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research code release accompanying the NeurIPS 2025 paper:**

> Moritz Haas, Sebastian Bordt, Ulrike von Luxburg, and Leena Chennuru Vankadara. "On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling." *arXiv preprint arXiv:2505.22491*, 2025.

---

## ⚠️ Important Notice

**This is research code, not production software.** It is designed for:
- Researchers monitoring statistics of **small to medium-sized neural networks**
- Performing **refined coordinate checks** for hyperparameter tuning analysis
- **Single-GPU training** scenarios

**Not suitable for:**
- Production deployments
- Multi-GPU training (see [Multi-GPU Limitations](#multi-gpu-training))
- Large-scale models without careful memory management

---

## Overview

`torch-module-monitor` enables deep-dive diagnostics of PyTorch model training with minimal code changes. The library provides two main capabilities:

### 1. Arbitrary Metric Computation
Monitor **activations**, **parameters**, and **gradients** across all model layers:
- Define custom metrics with a single line of code
- Apply metrics selectively using regex-based layer filtering
- Built-in support for monitoring attention mechanisms (query/key/value tensors and entropy)
- Aggregate statistics across micro-batches
- Export metrics to HDF5 for post-training analysis

### 2. Refined Coordinate Check (RCC)
Implementation of the refined muP coordinate check from our paper:
- Decomposes activation changes into: **(W_t - W_0)x_t** (weight updates) vs. **W_0(x_t - x_0)** (input drift)
- Essential for analyzing training dynamics under different hyperparameter scalings
- Requires reference module (e.g., model at initialization)

---

## Installation

```bash
pip install torch-module-monitor
```

**Development install:**
```bash
git clone https://github.com/tml-tuebingen/torch-module-monitor.git
cd torch-module-monitor
pip install -e .
```

**Optional dependencies:**
```bash
pip install torch-module-monitor[wandb]  # Weights & Biases integration
pip install torch-module-monitor[tb]     # TensorBoard integration
pip install torch-module-monitor[dev]    # Development tools (pytest, black, etc.)
```

---

## Quick Start

### Basic Monitoring

```python
from torch_module_monitor import ModuleMonitor

# Initialize monitor
monitor = ModuleMonitor(
    monitor_step_fn=lambda step: step % 10 == 0  # Monitor every 10 steps
)
monitor.set_module(model)

# Define metrics in one line each
monitor.add_activation_metric("mean", lambda x: x.mean())
monitor.add_activation_metric("std", lambda x: x.std())
monitor.add_parameter_metric("norm", lambda x: x.norm())
monitor.add_gradient_metric("norm", lambda x: x.norm())

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    monitor.begin_step(step)

    outputs = model(inputs)  # Activations captured automatically via hooks
    loss = criterion(outputs, targets)
    loss.backward()

    monitor.monitor_parameters()  # Compute parameter metrics
    monitor.monitor_gradients()   # Compute gradient metrics

    optimizer.step()
    optimizer.zero_grad()
    monitor.end_step()

    # Retrieve metrics for logging
    if monitor.is_step_monitored(step):
        metrics = monitor.get_step_metrics()
        wandb.log(metrics)  # or tensorboard, etc.
```

### Refined Coordinate Check

```python
import copy
from torch_module_monitor import ModuleMonitor, RefinedCoordinateCheck

# Save model at initialization
model_init = copy.deepcopy(model)

# Set up monitoring with reference module
monitor = ModuleMonitor(monitor_step_fn=lambda step: step % 100 == 0)
monitor.set_module(model)
monitor.set_reference_module(model_init)

# Initialize RCC
rcc = RefinedCoordinateCheck(monitor)

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    monitor.begin_step(step)

    # Run reference model (not monitored, just stores activations)
    with monitor.no_monitor():
        _ = model_init(inputs)

    # Run trained model (monitored + compared with reference)
    outputs = model(inputs)

    # Perform refined coordinate check
    rcc.refined_coordinate_check()  # Logs (W_t-W_0)x_t and W_0(x_t-x_0)

    loss = criterion(outputs, targets)
    loss.backward()

    monitor.monitor_parameters()
    monitor.monitor_gradients()

    optimizer.step()
    optimizer.zero_grad()
    monitor.end_step()
```

The RCC logs metrics such as:
- `RCC (W_t-W_0)x_t/{layer_name}/l2norm` - Change from weight updates
- `RCC W_0(x_t-x_0)/{layer_name}/l2norm` - Change from input drift

---

## Key Features

### Regex-Based Layer Filtering

Apply metrics selectively to specific layers:

```python
# Monitor only attention layers
monitor.add_activation_metric(
    "attention_entropy",
    lambda x: compute_entropy(x),
    metric_regex=r".*attention.*"
)

# Monitor only output layers
monitor.add_parameter_metric(
    "output_norm",
    lambda x: x.norm(),
    metric_regex=r".*fc_out.*"
)
```

### Reference Module Comparison

Track parameter and activation drift from initialization:

```python
import copy

reference_model = copy.deepcopy(model)
monitor.set_reference_module(reference_model)

# Track how much parameters changed from initialization
monitor.add_parameter_difference_metric(
    "l2_distance",
    lambda param, ref_param: (param - ref_param).norm()
)

# Track how much activations changed from initialization
monitor.add_activation_difference_metric(
    "l2_distance",
    lambda act, ref_act: (act - ref_act).norm(dim=-1).mean()
)
```

### Micro-Batch Aggregation

For gradient accumulation scenarios:

```python
monitor.begin_step(step)

for micro_batch in batches:
    outputs = model(micro_batch)
    loss = criterion(outputs, targets)
    loss.backward()
    monitor.after_micro_batch()  # Clear reference activations

monitor.monitor_gradients()
optimizer.step()
monitor.end_step()  # Aggregates across all micro-batches
```

---

## Advanced Usage

### Complex Modules: MonitorMixin Pattern

**By default**, the monitor only tracks modules that return a single tensor. For complex modules with structured outputs (e.g., attention mechanisms), inherit from `MonitorMixin` to log custom metrics.

**Example: Monitoring Attention Mechanisms**

```python
from torch_module_monitor import MonitorMixin, monitor_scaled_dot_product_attention
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module, MonitorMixin):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        # ... initialize projections

    def forward(self, x):
        # Compute query, key, value
        q, k, v = self.compute_qkv(x)  # Shape: (batch, n_heads, seq_len, d_k)

        # Standard scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # Monitor attention internals (if monitoring is active)
        if self.is_monitoring:
            monitor_scaled_dot_product_attention(
                monitor=self.get_module_monitor(),
                module=self,
                query=q,
                key=k,
                value=v,
                activation=attn_output
            )

        return self.output_projection(attn_output)
```

This logs per-head metrics:
- `activation/{module_name}.head_{i}.query/metric_name`
- `activation/{module_name}.head_{i}.key/metric_name`
- `activation/{module_name}.head_{i}.value/metric_name`
- `attention_entropy/{module_name}.head_{i}` - Attention entropy

**Example: Custom Metrics in Modules**

```python
from torch_module_monitor import MonitorMixin

class CustomLayer(nn.Module, MonitorMixin):
    def forward(self, x):
        intermediate = self.transform(x)
        output = self.activation(intermediate)

        # Log custom statistics when monitoring is active
        if self.is_monitoring:
            monitor = self.get_module_monitor()
            monitor.log_tensor("intermediate_norm", intermediate.norm(dim=-1))
            monitor.log_scalar("sparsity", (intermediate == 0).float().mean())

        return output
```

### Export and Analysis

```python
# Export all metrics to HDF5
monitor.storage.save_hdf5("metrics.h5")

# Load for post-training analysis
from torch_module_monitor import StorageManager
metrics = StorageManager.load_hdf5("metrics.h5")

# Metrics are organized as: {metric_name: {step: value}}
for metric_name, step_values in metrics.items():
    steps = sorted(step_values.keys())
    values = [step_values[s] for s in steps]
    # Plot or analyze...
```

---

## Multi-GPU Training

**Current status:** The code was developed and tested for **single-GPU training**.

### What Works
- Activation, parameter, and gradient tracking on a single GPU

### Multi-GPU Limitations

**Activation tracking:**
- Each GPU would need its own `ModuleMonitor` instance
- Activations are distributed across GPUs (not automatically gathered)
- Extension to multi-GPU is straightforward but requires manual setup

**Parameter and gradient tracking:**
- Parameters and gradients are typically replicated or sharded across GPUs
- Would need to gather metrics on rank 0 for centralized logging
- Requires integration with your distributed training framework (DDP, FSDP, etc.)

**Refined Coordinate Check:**
- ⚠️ **Not straightforward to extend to multi-GPU**
- RCC requires additional forward passes with mixed inputs
- Synchronization across GPUs for reference activations is non-trivial
- We recommend using RCC on single-GPU or smaller models

### Recommendation
For multi-GPU scenarios, we recommend:
1. Use single-GPU for RCC analysis on a smaller proxy model
2. Monitor basic metrics (norms, means) on multi-GPU setups with custom gathering logic
3. Contribute multi-GPU support (PRs welcome!)

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{haas2025splargelr,
  title={On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling},
  author={Haas, Moritz and Bordt, Sebastian and von Luxburg, Ulrike and Vankadara, Leena Chennuru},
  journal={arXiv preprint arXiv:2505.22491},
  year={2025}
}
```

---

## Contributing

This is research code, but contributions are welcome! If you:
- Find bugs
- Add multi-GPU support
- Implement new monitoring patterns
- Improve documentation

Please open an issue or pull request on GitHub.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Developed at the [Tübingen Machine Learning Lab](https://tml.cs.uni-tuebingen.de/).
