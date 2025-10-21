# torch-module-monitor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research code release for monitoring PyTorch model training dynamics. Accompanies the NeurIPS 2025 paper ["On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling"](https://arxiv.org/abs/2505.22491) (Haas et al., 2025).

**Designed for:** Researchers monitoring small-to-medium neural networks on single-GPU setups.
**Not for:** Production deployments or multi-GPU training (see [limitations](#multi-gpu-training) below).

---

## Features

**1. Arbitrary Metric Computation**
- Monitor activations, parameters, and gradients with one line of code per metric
- Regex-based layer filtering for selective monitoring
- Built-in attention mechanism support (query/key/value, entropy)
- Micro-batch aggregation and HDF5 export

**2. Refined Coordinate Check (RCC)**
- Decomposes activation changes: **(W_t - W_0)x_t** (weight updates) vs. **W_0(x_t - x_0)** (input drift)
- Essential for analyzing hyperparameter scaling in muP/SP settings

---

## Installation

```bash
pip install torch-module-monitor
```

**Optional dependencies:**
```bash
pip install torch-module-monitor[wandb]  # Weights & Biases
pip install torch-module-monitor[tb]     # TensorBoard
pip install torch-module-monitor[dev]    # Development tools
```

---

## Quick Start

### Basic Monitoring

```python
from torch_module_monitor import ModuleMonitor

# Initialize and add metrics
monitor = ModuleMonitor(monitor_step_fn=lambda step: step % 10 == 0)
monitor.set_module(model)

monitor.add_activation_metric("mean", lambda x: x.mean())
monitor.add_parameter_metric("norm", lambda x: x.norm())
monitor.add_gradient_metric("norm", lambda x: x.norm())

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    monitor.begin_step(step)

    outputs = model(inputs)  # Activations captured via hooks
    loss = criterion(outputs, targets)
    loss.backward()

    monitor.monitor_parameters()
    monitor.monitor_gradients()

    optimizer.step()
    optimizer.zero_grad()
    monitor.end_step()

    # Log metrics
    if monitor.is_step_monitored(step):
        wandb.log(monitor.get_step_metrics())
```

### Refined Coordinate Check

```python
import copy
from torch_module_monitor import ModuleMonitor, RefinedCoordinateCheck

# Save model at initialization
model_init = copy.deepcopy(model)

monitor = ModuleMonitor(monitor_step_fn=lambda step: step % 100 == 0)
monitor.set_module(model)
monitor.set_reference_module(model_init)

rcc = RefinedCoordinateCheck(monitor)

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    monitor.begin_step(step)

    with monitor.no_monitor():
        _ = model_init(inputs)  # Store reference activations
    outputs = model(inputs)

    rcc.refined_coordinate_check()  # Compute (W_t-W_0)x_t and W_0(x_t-x_0)

    # ... backward, step, end_step as above
```

**See [examples/](examples/) for complete examples:**
- `metrics.ipynb` - Basic metric monitoring
- `reference-model.ipynb` - Reference module comparison
- `refined-coordinate-check.ipynb` - Full RCC workflow

**Real-world integration:** See our [monitored nanoGPT](https://github.com/tml-tuebingen/nanoGPT-monitored) for a complete transformer training example.

---

## Key Patterns

### Regex-Based Layer Filtering

```python
# Monitor only attention layers
monitor.add_activation_metric(
    "entropy", lambda x: compute_entropy(x), metric_regex=r".*attn.*"
)
```

### Reference Module Comparison

```python
reference_model = copy.deepcopy(model)
monitor.set_reference_module(reference_model)

# Track drift from initialization
monitor.add_parameter_difference_metric(
    "l2_distance", lambda p, p_ref: (p - p_ref).norm()
)
```

### Complex Modules (MonitorMixin)

By default, only modules returning a single tensor are monitored. For complex outputs (e.g., attention), use `MonitorMixin`:

```python
from torch_module_monitor import MonitorMixin, monitor_scaled_dot_product_attention

class MultiHeadAttention(nn.Module, MonitorMixin):
    def forward(self, x):
        q, k, v = self.compute_qkv(x)
        attn_output = F.scaled_dot_product_attention(q, k, v)

        if self.is_monitoring:
            monitor_scaled_dot_product_attention(
                self.get_module_monitor(), module=self,
                query=q, key=k, value=v, activation=attn_output
            )

        return self.output_projection(attn_output)
```

This logs per-head metrics: `activation/{module}.head_{i}.query`, `attention_entropy/{module}.head_{i}`, etc.

**Custom metrics in any module:**

```python
from torch_module_monitor import MonitorMixin

class CustomLayer(nn.Module, MonitorMixin):
    def forward(self, x):
        output = self.transform(x)

        if self.is_monitoring:
            self.get_module_monitor().log_tensor("custom_stat", output.norm(dim=-1))

        return output
```

### Export to HDF5

```python
monitor.storage.save_hdf5("metrics.h5")

# Load for analysis
from torch_module_monitor import StorageManager
metrics = StorageManager.load_hdf5("metrics.h5")  # {metric_name: {step: value}}
```

---

## Multi-GPU Training

**Current support:** Single-GPU only. The code was developed and tested for single-GPU training.

**Limitations:**
- **Activation tracking:** Each GPU needs its own monitor; activations not gathered automatically
- **Parameters/gradients:** Would require gathering on rank 0 for centralized logging
- **Refined Coordinate Check:** Not straightforward to extend to multi-GPU due to synchronization requirements

**Recommendation:** Use single-GPU for RCC analysis (possibly on a smaller proxy model), or contribute multi-GPU support!

---

## Citation

If you use this code, please cite:

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

Contributions welcome! This is research code, but we appreciate bug reports, multi-GPU support, new monitoring patterns, and documentation improvements.

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

Developed at the [Tübingen Machine Learning Lab](https://tml.cs.uni-tuebingen.de/).
