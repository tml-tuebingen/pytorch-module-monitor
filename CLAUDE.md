# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pytorch-module-monitor is a diagnostic library for monitoring PyTorch model training. It provides deep-dive diagnostics by tracking activations, parameters, and gradients across all layers during training. 

## Key Architecture Concepts

### Core Components

1. **ModuleMonitor** (`src/pytorch_module_monitor/monitor.py`): The main class that orchestrates all monitoring activities
   - Registers forward hooks on all submodules to capture activations
   - Supports comparing a trained model against a reference model (e.g., model at initialization)
   - Uses a step-based monitoring system to selectively monitor specific training steps
   - Implements a metric system with regex-based filtering for flexible monitoring

2. **StorageManager** (`src/pytorch_module_monitor/storage.py`): Handles metric storage and aggregation
   - Stores metrics in a nested dictionary structure: `{step: {metric_key: value}}`
   - Supports both scalar and tensor logging
   - Aggregates metrics across micro-batches using custom or default aggregation functions
   - Can export to HDF5 format for analysis

3. **HooksManager** (`src/pytorch_module_monitor/hooks.py`): Manages PyTorch forward hooks
   - Provides automatic cleanup of hooks
   - Tracks all registered hooks for easy removal

4. **MonitorMixin** (`src/pytorch_module_monitor/monitor.py`): Interface for custom module monitoring
   - Subclass this in custom modules to log additional metrics during forward pass
   - Automatically detected and initialized by ModuleMonitor

### Advanced Features

1. **Reference Module Comparison**: Compare activations/parameters against a reference model (e.g., at initialization)
   - Requires running reference module forward pass before monitored module
   - Stores reference activations for comparison
   - Supports activation difference metrics

2. **Refined Coordinate Check (RCC)** (`src/pytorch_module_monitor/refined_coordinate_check.py`):
   - Implements the refined muP coordinate check from https://arxiv.org/abs/2505.22491
   - Decomposes activation changes into weight changes vs input changes: `(W_t-W_0)x_t` vs `W_0(x_t-x_0)`
   - Requires both a monitored module and a reference module
   - Performs additional forward passes to compute metrics

3. **Attention Monitoring** (`src/pytorch_module_monitor/attention.py`):
   - Specialized function for monitoring scaled dot-product attention operations
   - Monitors per-head query/key/value tensors and attention entropy
   - Must be called manually within attention modules

### Metric System

Metrics are defined with three components:
- **metric_fn**: Function that computes the metric from tensors
- **metric_regex**: Regex pattern matching module/parameter names to monitor
- **metric_aggregation_fn**: (Optional) Custom aggregation across micro-batches

Metric types:
- **Activation metrics**: Applied to module outputs during forward pass
- **Activation difference metrics**: Compare activations with reference module
- **Parameter metrics**: Applied to model parameters
- **Parameter difference metrics**: Compare parameters with reference module
- **Gradient metrics**: Applied to parameter gradients after backward pass

### Monitoring Workflow

The typical training loop integration:

```python
monitor.begin_step(step)                 # Start monitoring step
output = model(input)                     # Forward pass (hooks capture activations)
loss.backward()                           # Backward pass
monitor.monitor_parameters()              # Log parameter metrics
monitor.monitor_gradients()               # Log gradient metrics
optimizer.step()
monitor.end_step()                        # Aggregate metrics for this step
metrics = monitor.get_step_metrics()      # Retrieve metrics for logging
```

For micro-batching:
```python
monitor.begin_step(step)
for micro_batch in batches:
    output = model(micro_batch)
    # ... backward, gradient accumulation
    monitor.after_micro_batch()           # Clear reference activations
monitor.end_step()                        # Aggregates across all micro-batches
```

## Development Commands

### Installation
```bash
# Install package in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[wandb]"     # For Weights & Biases integration
pip install -e ".[tb]"         # For TensorBoard integration
pip install -e ".[dev]"        # For development tools
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Testing
```bash
# Run tests (if tests directory exists)
pytest

# Run tests in parallel
pytest -n auto
```

## Important Implementation Details

1. **Tensor Detachment**: All captured tensors are detached and cloned to avoid memory leaks and computational graph issues (see monitor.py:439)

2. **CPU Offload**: The `cpu_offload` parameter moves captured activations to CPU to save GPU memory (expensive but necessary for large models)

3. **Module Name Formatting**: The `format_module_name_fn` normalizes module names (handles FSDP, torch.compile wrappers, etc.)

4. **Excluded Modules**: Use `excluded_modules_regex` to skip monitoring certain modules (e.g., modules called multiple times per forward pass like pooling layers)

5. **Reference Module Structure Validation**: The reference module must have identical structure to the monitored module (same module names)

6. **RCC Additional Forward Passes**: `refined_coordinate_check()` performs extra forward passes with mixed inputs, so wrap it properly to avoid interfering with training

7. **Storage Size**: The logger reports storage size after each step aggregation - monitor this for memory usage

## Integration with Logging Frameworks

The library is framework-agnostic. Get metrics with `get_step_metrics()` and log to your preferred backend:

```python
metrics = monitor.get_step_metrics()
wandb.log(metrics)              # Weights & Biases
tensorboard.add_scalars(metrics)  # TensorBoard
```

## Common Patterns

- **Selective monitoring**: Use `monitor_step_fn` to monitor only certain steps (e.g., every 100 steps)
- **Per-layer metrics**: Use regex patterns to apply different metrics to different layer types
- **Custom metrics in modules**: Subclass `MonitorMixin` and call `self.get_module_monitor().log_tensor()` during forward pass
