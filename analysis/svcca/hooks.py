from contextlib import contextmanager
from typing import Any
from functools import partial

from torch import nn
import torch
from transformers.modeling_utils import PreTrainedModel


@torch.inference_mode()
@contextmanager
def collect_activations(
    model: PreTrainedModel,
    hookpoints: list[str],
    token: int | None = None,
    input_acts: bool = False,
):
    """
    Context manager that hooks a model and collects activations.
    An activation tensor is produced for each batch processed and stored
    in added to a list for that hookpoint in the activations dictionary.
    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to collect activations from
        input_acts: Whether to collect input activations or output activations
    Yields:
        Dictionary mapping hookpoints to their collected activations
    """
    activations = {}
    handles = []

    def create_input_hook(hookpoint: str):
        def input_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(input, tuple):
                activations[hookpoint] = input[0].detach().cpu()
            else:
                activations[hookpoint] = input.detach().cpu()

            if token != None:
                activations[hookpoint] = activations[hookpoint][:, token]

        return input_hook

    def create_output_hook(hookpoint: str):
        def output_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                activations[hookpoint] = output[0].detach().cpu()
            else:
                activations[hookpoint] = output.detach().cpu()

            if token != -1:
                activations[hookpoint] = activations[hookpoint][:, token]

        return output_hook

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            hook = create_input_hook(name) if input_acts else create_output_hook(name)
            handle = module.register_forward_hook(hook)
            handles.append(handle)

    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()

        handles.clear()


collect_input_activations = partial(collect_activations, input_acts=True)
collect_output_activations = partial(collect_activations, input_acts=False)

