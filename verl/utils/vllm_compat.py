"""Small compatibility helpers that do not import vLLM at module import time."""


def get_vllm_tensor_model_parallel_process_group(parallel_state):
    """Return the torch TP process group across old and current vLLM APIs."""
    get_group = getattr(parallel_state, "get_tp_group", None)
    if get_group is None:
        get_group = parallel_state.get_tensor_model_parallel_group
    group = get_group()
    return getattr(group, "device_group", group)
