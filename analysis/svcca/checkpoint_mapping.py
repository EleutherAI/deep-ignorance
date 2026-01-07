#!/usr/bin/env python3
"""
Map activations from early checkpoint to late checkpoint using SVCCA.
Analyzes how much of the representation change is due to rotation.
"""

from typing import Any
import math
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
import logging

# Local imports
from analysis.svcca.distance import svcca_transform
from analysis.svcca.svcca import (
    collect_module_activations,
    load_and_tokenize,
    extract_layer_idx,
    module_group_map,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def compute_cosine_similarity_per_token(
    acts1: torch.Tensor,
    acts2: torch.Tensor
) -> tuple[float, float]:
    """
    Compute cosine similarity per token, return mean and std.

    Args:
        acts1: [num_tokens, features]
        acts2: [num_tokens, features]

    Returns:
        (mean, std) of cosine similarities across tokens
    """
    # Normalize
    acts1_norm = acts1 / (acts1.norm(dim=1, keepdim=True) + 1e-8)
    acts2_norm = acts2 / (acts2.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity per token
    cos_sims = (acts1_norm * acts2_norm).sum(dim=1)

    return cos_sims.mean().item(), cos_sims.std().item()


def compute_svcca_mapping_for_module(
    module: str,
    gpu_id: int,
    early_activations: list[dict[str, torch.Tensor]],
    late_activations: list[dict[str, torch.Tensor]],
) -> tuple[str, dict]:
    """
    Compute SVCCA transformation and cosine similarities for a module.

    Returns:
        (module_name, results_dict)
    """
    device = f"cuda:{gpu_id}"

    # Concatenate activations from all batches
    early_acts = torch.cat(
        [item[module].to(device, non_blocking=True).flatten(0, 1)
         for item in early_activations],
        dim=0,
    ).to(torch.float32)

    late_acts = torch.cat(
        [item[module].to(device, non_blocking=True).flatten(0, 1)
         for item in late_activations],
        dim=0,
    ).to(torch.float32)

    # Compute baseline cosine similarity (original)
    cos_orig_mean, cos_orig_std = compute_cosine_similarity_per_token(
        early_acts, late_acts
    )

    # Compute SVCCA transformation
    x_reduced, y_reduced, a, b, diag = svcca_transform(
        early_acts, late_acts, 0.99, "svd"
    )

    # Transform early activations using SVCCA mapping
    # Project to CCA space
    early_in_cca = x_reduced @ a
    late_in_cca = y_reduced @ b

    # Project back to late's original space
    # We transform: early_cca @ b.T gives us coordinates in y_reduced space
    # Then @ y_reduced.T projects back to original late space
    transformed_early = early_in_cca @ b.T @ y_reduced.T

    # Compute cosine similarity after transformation
    cos_trans_mean, cos_trans_std = compute_cosine_similarity_per_token(
        transformed_early, late_acts
    )

    # SVCCA similarity score
    div = min(a.size(1), b.size(1))
    svcca_sim = 1.0 - (1.0 - diag.sum() / div).item()

    results = {
        "cosine_original_mean": cos_orig_mean,
        "cosine_original_std": cos_orig_std,
        "cosine_transformed_mean": cos_trans_mean,
        "cosine_transformed_std": cos_trans_std,
        "cosine_improvement": cos_trans_mean - cos_orig_mean,
        "svcca_similarity": svcca_sim,
        "transformation_rank": div,
    }

    logger.info(
        f"Module {module}: orig={cos_orig_mean:.4f}, trans={cos_trans_mean:.4f}, "
        f"improvement={results['cosine_improvement']:.4f}"
    )

    # Free GPU memory
    del early_acts, late_acts, x_reduced, y_reduced, a, b
    del early_in_cca, late_in_cca, transformed_early
    torch.cuda.empty_cache()

    return module, results


def analyze_checkpoint_mapping(
    early_model: str,
    early_revision: str,
    late_model: str,
    dataset: Dataset,
    batch_size: int,
    module_batch_size: int,
    target_layers: list[int] | None,
    num_gpus: int,
    tokens_per_sequence: int,
    sample_strategy: str,
    max_modules: int | None = None,
) -> dict[str, dict]:
    """
    Main analysis function that orchestrates checkpoint comparison.

    Args:
        early_model: HuggingFace model name for early checkpoint
        early_revision: Git revision/checkpoint name for early model
        late_model: HuggingFace model name for late checkpoint
        dataset: Tokenized dataset to collect activations from
        batch_size: Batch size for activation collection
        module_batch_size: Number of modules to process at once (for memory)
        target_layers: List of layer indices to analyze (None = all layers)
        num_gpus: Number of GPUs for parallel SVCCA computation
        tokens_per_sequence: Number of tokens to sample per sequence
        sample_strategy: Token sampling strategy ("space_evenly" or "end")

    Returns:
        Dictionary mapping module names to their analysis results
    """
    logger.info("Loading model to identify target modules...")

    # Get target modules (reuse from svcca.py)
    model = AutoModelForCausalLM.from_pretrained(late_model)
    named_modules = model.base_model.named_modules()

    target_module_info = {}
    for name, module in named_modules:
        if isinstance(module, nn.Linear):
            layer = extract_layer_idx(name)
            if layer is None:
                continue

            if target_layers is not None and layer not in target_layers:
                continue

            group = module_group_map(name)
            if group == "other":
                continue

            target_module_info[name] = {
                "layer": layer,
                "group": group,
            }

    del model
    torch.cuda.empty_cache()

    logger.info(f"Identified {len(target_module_info)} target modules")
    
    # Limit to max_modules if specified
    if max_modules is not None and len(target_module_info) > max_modules:
        module_names = list(target_module_info.keys())[:max_modules]
        target_module_info = {name: target_module_info[name] for name in module_names}
        logger.info(f"Limited to {len(target_module_info)} modules (max_modules={max_modules})")

    # Collect activations in batches
    module_batches = [
        list(target_module_info.keys())[i : i + module_batch_size]
        for i in range(0, len(target_module_info), module_batch_size)
    ]

    logger.info(f"Processing {len(module_batches)} module batches")

    for batch_idx, modules_batch in enumerate(module_batches):
        logger.info(f"Processing module batch {batch_idx + 1}/{len(module_batches)}")

        # Load early checkpoint
        logger.info(f"Loading early checkpoint: {early_model}@{early_revision}")
        early_model_obj = AutoModelForCausalLM.from_pretrained(
            early_model, revision=early_revision, device_map="auto"
        )
        early_acts = collect_module_activations(
            early_model_obj,
            f"{early_model}@{early_revision}",
            modules_batch,
            dataset,
            batch_size=batch_size,
            tokens_per_sequence=tokens_per_sequence,
            sample_strategy=sample_strategy,
        )
        del early_model_obj
        torch.cuda.empty_cache()

        # Load late checkpoint
        logger.info(f"Loading late checkpoint: {late_model}")
        late_model_obj = AutoModelForCausalLM.from_pretrained(
            late_model, device_map="auto"
        )
        late_acts = collect_module_activations(
            late_model_obj,
            late_model,
            modules_batch,
            dataset,
            batch_size=batch_size,
            tokens_per_sequence=tokens_per_sequence,
            sample_strategy=sample_strategy,
        )
        del late_model_obj
        torch.cuda.empty_cache()

        # Parallel SVCCA computation
        logger.info(f"Computing SVCCA mappings across {num_gpus} GPUs...")
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for idx, module in enumerate(modules_batch):
                gpu_id = idx % num_gpus
                future = executor.submit(
                    compute_svcca_mapping_for_module,
                    module,
                    gpu_id,
                    early_acts,
                    late_acts,
                )
                futures[future] = module

            # Collect results
            with tqdm(total=len(modules_batch), desc="Computing SVCCA mappings") as pbar:
                for future in futures:
                    module, results = future.result()
                    target_module_info[module].update(results)
                    pbar.update(1)

    return target_module_info
