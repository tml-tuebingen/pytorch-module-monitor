# Testing Plan for pytorch-module-monitor

## Test Structure Overview

```
tests/
├── conftest.py                      # Shared fixtures and test utilities
├── test_storage.py                  # StorageManager tests
├── test_hooks.py                    # HooksManager tests
├── test_monitor_basic.py            # Basic ModuleMonitor functionality
├── test_monitor_metrics.py          # Metric system tests
├── test_monitor_reference.py        # Reference module comparison tests
├── test_monitor_mixin.py            # MonitorMixin interface tests
├── test_refined_coordinate_check.py # RCC tests
├── test_attention.py                # Attention monitoring tests
├── test_integration.py              # End-to-end integration tests
└── test_edge_cases.py               # Error handling and edge cases
```

## 1. Test Fixtures (conftest.py)

**Purpose**: Provide reusable test models and utilities

### Simple models
- `simple_linear_model`: 2-3 layer MLP for basic tests
- `conv_model`: Simple CNN for testing convolutional layers
- `model_with_mixin`: Model with MonitorMixin modules
- `attention_model`: Model with attention layers

### Test data
- `sample_batch`: Small batch of random tensors
- `device_fixture`: Auto-detect CUDA availability

### Monitor fixtures
- `basic_monitor`: Pre-configured ModuleMonitor
- `monitor_with_metrics`: Monitor with common metrics registered

## 2. StorageManager Tests (test_storage.py)

**Unit tests for storage and aggregation**

### Basic operations
- `test_create_step_entry`: Creating step entries
- `test_log_scalar`: Logging single scalars
- `test_log_scalar_multiple`: Logging multiple values for same key (list accumulation)
- `test_log_tensor`: Logging tensors

### Aggregation
- `test_aggregate_step_scalars`: Default mean aggregation for scalars
- `test_aggregate_step_tensors`: Default mean aggregation for tensors
- `test_custom_aggregation_fn`: Custom aggregation functions
- `test_aggregation_dict_output`: Aggregation functions returning dicts (e.g., {'[mean]': x, '[std]': y})

### Data retrieval
- `test_get_step_metrics`: Retrieve specific step metrics
- `test_get_all_metrics`: Retrieve all logged metrics
- `test_condensed_log_dict`: Transform to condensed format

### Persistence
- `test_save_load_hdf5`: Save and load HDF5 files
- `test_hdf5_entry_keys`: Read entry keys from HDF5

## 3. HooksManager Tests (test_hooks.py)

**Unit tests for hook management**

- `test_register_forward_hook`: Register hooks on modules
- `test_remove_hook`: Remove specific hook
- `test_remove_all_hooks`: Remove all hooks
- `test_hook_replacement`: Re-registering hook on same module
- `test_hook_execution`: Verify hooks are actually called during forward pass
- `test_context_manager`: Test context manager cleanup
- `test_destructor_cleanup`: Verify __del__ removes hooks

## 4. Basic ModuleMonitor Tests (test_monitor_basic.py)

**Core monitoring functionality**

### Setup and initialization
- `test_monitor_init`: Basic initialization
- `test_set_module`: Setting module to monitor
- `test_set_module_twice_fails`: Cannot change module after setting
- `test_hooks_registered`: Verify hooks are registered on all submodules

### Step management
- `test_begin_step`: Start monitoring step
- `test_end_step`: End monitoring step and aggregate
- `test_is_step_monitored`: Check if step should be monitored
- `test_after_micro_batch`: Cleanup after micro-batch
- `test_no_monitor_context`: Temporary disable monitoring

### Module name formatting
- `test_default_format_module_name`: Default formatting (handles _orig_mod, etc.)
- `test_custom_format_module_name`: Custom formatting function
- `test_excluded_modules_regex`: Module exclusion by regex

## 5. Metric System Tests (test_monitor_metrics.py)

**Test metric registration and computation**

### Activation metrics
- `test_add_activation_metric`: Register activation metric
- `test_activation_metric_computation`: Verify metric is computed during forward pass
- `test_activation_metric_regex`: Regex filtering for specific modules
- `test_activation_metric_aggregation`: Custom aggregation across batches

### Parameter metrics
- `test_add_parameter_metric`: Register parameter metric
- `test_monitor_parameters`: Compute parameter metrics
- `test_parameter_metric_regex`: Regex filtering

### Gradient metrics
- `test_add_gradient_metric`: Register gradient metric
- `test_monitor_gradients`: Compute gradient metrics after backward
- `test_gradient_before_clip`: Test before_clip flag
- `test_gradient_none_handling`: Handle parameters with None gradients

### Manual logging
- `test_log_scalar`: Manual scalar logging
- `test_log_scalar_force`: Force logging outside monitoring window
- `test_log_scalars_dict`: Log dictionary of scalars
- `test_log_tensor`: Manual tensor logging

## 6. Reference Module Tests (test_monitor_reference.py)

**Test reference module comparison**

### Setup
- `test_set_reference_module`: Set reference module
- `test_reference_module_structure_validation`: Verify structure matches
- `test_remove_reference_module`: Remove reference module
- `test_has_reference_module`: Check if reference module is set

### Activation comparison
- `test_reference_activation_storage`: Reference activations are stored
- `test_activation_difference_metric`: Compute activation differences
- `test_reference_forward_before_monitored`: Reference must run first

### Parameter comparison
- `test_parameter_difference_metric`: Compute parameter differences
- `test_parameter_difference_missing_param`: Handle missing parameters

### CPU offload
- `test_cpu_offload_reference_activations`: Test CPU offload functionality

## 7. MonitorMixin Tests (test_monitor_mixin.py)

**Test custom module monitoring interface**

- `test_mixin_initialization`: MonitorMixin is initialized on subclassed modules
- `test_mixin_is_monitoring`: Check monitoring status from module
- `test_mixin_get_module_monitor`: Access monitor from module
- `test_mixin_custom_logging`: Custom logging from within module
- `test_mixin_reference_flag`: is_reference_module flag is set correctly

## 8. RefinedCoordinateCheck Tests (test_refined_coordinate_check.py)

**Test RCC functionality**

### Setup
- `test_rcc_init_requires_module`: Error if module not set
- `test_rcc_init_requires_reference`: Error if reference module not set
- `test_rcc_hooks_registered`: Verify hooks are registered

### Coordinate check computation
- `test_rcc_linear_layer`: Test RCC on linear layer
- `test_rcc_layer_norm`: Test RCC on layer norm
- `test_rcc_embedding`: Test RCC on embedding layer
- `test_rcc_weight_decomposition`: Verify (W_t-W_0)x_t and W_0(x_t-x_0) computation
- `test_rcc_bias_handling`: Test bias-free metrics for linear/layernorm

### Edge cases
- `test_rcc_no_forward_pass_error`: Error if called before forward pass
- `test_rcc_excluded_modules`: Excluded modules are skipped

## 9. Attention Monitoring Tests (test_attention.py)

**Test attention-specific monitoring**

- `test_monitor_scaled_dot_product_attention`: Basic attention monitoring
- `test_attention_per_head_metrics`: Per-head query/key/value monitoring
- `test_attention_entropy`: Attention entropy computation
- `test_attention_with_mask`: Attention with mask
- `test_attention_causal`: Causal attention
- `test_attention_gqa`: Grouped query attention

## 10. Integration Tests (test_integration.py)

**End-to-end workflows**

- `test_full_training_loop`: Complete training loop with monitoring
- `test_micro_batch_accumulation`: Multiple micro-batches per step
- `test_reference_module_workflow`: Full workflow with reference module
- `test_rcc_workflow`: Full workflow with RCC
- `test_monitoring_disabled_steps`: Verify no overhead when not monitoring
- `test_compiled_model`: Test with torch.compile() wrapper
- `test_fsdp_model`: Test with FSDP wrapper (if available)

## 11. Edge Cases and Error Handling (test_edge_cases.py)

**Robustness tests**

### Error conditions
- `test_module_not_set_error`: Error when monitoring without setting module
- `test_step_backward_error`: Error when step goes backward
- `test_metric_duplicate_name_error`: Error on duplicate metric names
- `test_non_tensor_activation_handling`: Skip non-tensor activations

### Edge cases
- `test_empty_model`: Model with no submodules
- `test_module_called_multiple_times`: Modules called multiple times per forward (should warn)
- `test_very_large_batch`: Memory handling with large batches
- `test_nan_inf_handling`: Handle NaN/Inf in metrics

### Metric function errors
- `test_metric_function_exception`: Graceful handling of metric function errors

## Test Coverage Goals

- **Target**: >90% code coverage
- **Focus areas**:
  - All public methods
  - Error paths
  - Edge cases with tensor shapes
  - Different module types (Linear, Conv2d, LayerNorm, Embedding, etc.)

## Testing Best Practices

1. **Use parametrize**: Test multiple scenarios with `@pytest.mark.parametrize`
2. **Mock when needed**: Mock expensive operations (e.g., large model forwards)
3. **Test both CPU and GPU**: If CUDA available, test on both devices
4. **Clear state**: Ensure tests clean up hooks and don't leak memory
5. **Descriptive names**: Test names clearly indicate what's being tested
6. **Assert messages**: Include helpful messages in assertions

## Implementation Order

Recommended order for implementing tests:

1. **conftest.py** - Set up fixtures first
2. **test_storage.py** - Foundation for all other tests
3. **test_hooks.py** - Simple unit tests
4. **test_monitor_basic.py** - Core functionality
5. **test_monitor_metrics.py** - Build on basic functionality
6. **test_monitor_reference.py** - Reference module features
7. **test_monitor_mixin.py** - MonitorMixin interface
8. **test_refined_coordinate_check.py** - Advanced RCC features
9. **test_attention.py** - Specialized attention monitoring
10. **test_integration.py** - End-to-end scenarios
11. **test_edge_cases.py** - Error handling and edge cases

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_storage.py

# Run with coverage
pytest --cov=pytorch_module_monitor --cov-report=html

# Run in parallel
pytest -n auto

# Run with verbose output
pytest -v
```
