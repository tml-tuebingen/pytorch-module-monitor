# Monitor the training of your PyTorch modules

![monitor](images/landing.png)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a research codebase to monitor the training dynamics of small-to-medium neural networks. Log arbitrary metrics of activations, gradients, and parameters to Weights & Biases with a few lines of code!

We also provide an implementation of the refined coordinate check (RCC) from the NeurIPS 2025 paper ["On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling"](https://arxiv.org/abs/2505.22491) (Haas et al., 2025).

⚡For a complete working example, see how the monitor can be integrated into [nanoGPT](https://github.com/tml-tuebingen/nanoGPT-monitored).⚡


### Installation

```bash
pip install torch-module-monitor
```

---

## Features

**1. Monitor arbitrary metrics of activations, gradients, and parameters with a few lines of code**
- Add new metrics for activations, gradients, and parameters with a single line of code.
- Regex-based filtering to determine what should be logged
- Monitor the internals of the attention operation (query/key/value tensor metrics, attention entropy)
- Aggregation of activation metrics across micro-batches

**2. Perform the Refined Coordinate Check (RCC) from https://arxiv.org/abs/2505.22491**
- We provide an implementation of the refined coordinate check.

---

## Basic Monitoring

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

## Complete Examples

**See [examples/](examples/) for complete examples:**
- `metrics.ipynb` - Basic metric monitoring
- `reference-model.ipynb` - Reference module comparison
- `refined-coordinate-check.ipynb` - Refined coordinate check

⚡We also show how to integrate the monitor into [nanoGPT](https://github.com/tml-tuebingen/nanoGPT-monitored).⚡


---

## Integration with Weights & Biases

We name the different metrics such that they are nicely visualized in Weights & Biases.

Log the collected metrics in a single line of code:

```python
wandb.log(training_monitor.get_step_metrics(), step=current_step)
```

[Link to wandb logs of nanoGPT run](https://wandb.ai/public-runs/shakespeare-char/runs/943n8dnt)

![monitor](images/wandb.png)

---

## Patterns

### Regex-Based Module Filtering

You can use a regex to specify that a metric should only be computed for specific tensors.

```python
# Monitor only attention layers
monitor.add_activation_metric(
    "my_metric", my_metric(x), metric_regex=r".*mlp.*"
)
```

### Reference Module Comparison

In infinite width theory, we often want to measure the difference of activations and parameters to the model at initialization. We implement this via an arbitrary reference model to which our model can be compared.

```python
monitor.set_reference_module(reference_model)

# Track drift from initialization
monitor.add_parameter_difference_metric(
    "l2_distance", lambda p, p_ref: (p - p_ref).norm()
)
```

## Complex Modules

By default, we monitor the activations of modules that return a single tensor. To monitor statistics of complex modules, these modules can implement `MonitorMixin`. We use this approach to monitor the internals of the attention operation. 

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

## Multi-GPU Support

In principle, the monitor can support multi-GPU training, though we do not provide direct support for any parallelization strategy. With FSDP, for example, every GPU could have its own monitor. However, we do not currently implement the synchronization of activation metrics across GPUs. The refined coordinate check was only tested for single-GPU training. 

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{haas2025splargelr,
  title={On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling},
  author={Haas, Moritz and Bordt, Sebastian and von Luxburg, Ulrike and Vankadara, Leena Chennuru},
  booktitle={Advances in Neural Information Processing Systems 38},
  year={2025}
}
```

## Contributing

We provide this code as-is. We may accept pull requests that fix bugs or add new features. 


