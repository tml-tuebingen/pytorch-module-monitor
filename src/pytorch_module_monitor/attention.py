#################################################################
# Monitor the activations inside a scaled dot product attention operation
# This function has to be called by the monitored module
#################################################################

import functools
from typing import Callable, Any, Union
import torch
import math
from .monitor import ModuleMonitor

def monitor_scaled_dot_product_attention(
    monitor: ModuleMonitor,
    module: Union[str, torch.nn.Module],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    activation: torch.Tensor = None,
    is_reference: bool = False
) -> None:
    """Monitor query, key, value tensors and attention entropy in an attention operation.

    Must be called manually within attention modules. Logs per-head metrics for:
    - Query/key/value activations
    - Output activations (if provided)
    - Attention entropy

    Args:
        monitor: The ModuleMonitor instance.
        module: Module name or module instance for logging.
        query: Query tensor with shape (..., n_heads, seq_len, head_dim).
        key: Key tensor with shape (..., n_heads, seq_len, head_dim).
        value: Value tensor with shape (..., n_heads, seq_len, head_dim).
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability for attention weights.
        is_causal: Whether to use causal masking.
        scale: Optional scale factor (default: 1/sqrt(head_dim)).
        enable_gqa: Enable grouped query attention (repeats keys/values).
        activation: Optional output activation tensor to monitor.
        is_reference: Whether this is from the reference module.
    """
    if not monitor.is_monitoring():
        return
    
    module_name = monitor._get_module_name(module, is_reference)
    
    # Detach all tensors from the computational graph
    query = query.detach().clone()
    key = key.detach().clone()
    value = value.detach().clone()
    if activation is not None:
        activation = activation.detach().clone()
    if attn_mask is not None:
        attn_mask = attn_mask.detach().clone()
    
    # Compute attention weights following the reference implementation
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    
    # Monitor multi-head attention
    if S == L and query.size(-3) == key.size(-3) and query.size(-1) == value.size(-1):
        n_head = query.size(-3)
        for i_head in range(n_head):
            q = query[..., i_head, :, :]
            k = key[..., i_head, :, :]
            v = value[..., i_head, :, :]
            
            monitor.monitor_activations(f"{module_name}.head_{i_head}.query", q, is_reference)
            monitor.monitor_activations(f"{module_name}.head_{i_head}.key", k, is_reference)
            monitor.monitor_activations(f"{module_name}.head_{i_head}.value", v, is_reference)
            
            if activation is not None:
                o = activation[..., i_head, :, :]
                monitor.monitor_activations(f"{module_name}.head_{i_head}.activation", o, is_reference)
    else:
        monitor.logger.warning(
            f"Step {monitor.current_step}: monitor_scaled_dot_product_attention assumes "
            f"that S == L and that the key query and value tensor have the same dimension."
        )
    
    # Compute and monitor attention entropy
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    if S == L and query.size(-3) == key.size(-3) and query.size(-1) == value.size(-1):
        n_head = query.size(-3)
        for i_head in range(n_head):
            w = attn_weight[..., i_head, :, :]
            entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1)
            monitor.log_tensor(f"attention_entropy/{module_name}.head_{i_head}", entropy)
            monitor.logger.debug(
                f"Step {monitor.current_step}: Monitored attention entropy for head %s with shape %s",
                i_head, entropy.shape
            )
    
    monitor.logger.debug(
        f"Step {monitor.current_step}: Monitored scaled dot product attention for module %s "
        f"with query shape %s, key shape %s, value shape %s",
        module_name, query.shape, key.shape, value.shape
    )