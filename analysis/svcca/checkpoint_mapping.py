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

from torch import Tensor
import torch.nn.functional as F

# Local imports
from analysis.svcca.distance import cca
from analysis.svcca.svcca import (
    collect_module_activations,
    load_and_tokenize,
    extract_layer_idx,
    module_group_map,
)

# def set_deterministic():
#     torch.manual_seed(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)

# set_deterministic()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _svd_reduction_with_basis(x: Tensor, accept_rate: float) -> tuple[Tensor, Tensor]:
    """Returns (x_reduced, V_k) where x_reduced = x @ V_k"""
    # Assuming current implementation does something like:
    print("x", x.shape)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)

    # Compute cumulative variance ratio
    var_explained = S ** 2
    var_ratio = var_explained / var_explained.sum()
    cumulative_ratio = torch.cumsum(var_ratio, dim=0)
    
    # Find k: minimum components to reach accept_rate
    k = (cumulative_ratio < accept_rate).sum().item() + 1
    # print("using a k of", k, "to reach an accept_rate of", accept_rate)

    V_k = Vh[:k].T  # [original_dim, k]
    x_reduced = x @ V_k  # [n_samples, k]
    return x_reduced, V_k


def svcca_transform(
    x: Tensor,
    y: Tensor,
    accept_rate: float,
    backend: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Singular Vector CCA with transformation matrices.

    Similar to svcca_distance, but returns the transformation matrices
    and reduced representations for mapping between activation spaces.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>W
        accept_rate: 0.99 (threshold for SVD reduction)
        backend: svd or qr

    Returns:
        tuple of (x_reduced, y_reduced, a, b, diag) where:
        - x_reduced: SVD-reduced x (keeps components up to accept_rate variance)
        - y_reduced: SVD-reduced y (keeps components up to accept_rate variance)
        - a: CCA transformation matrix for x_reduced
        - b: CCA transformation matrix for y_reduced
        - diag: canonical correlations (diagonal of CCA)
    """
    x_reduced, V_k = _svd_reduction_with_basis(x, accept_rate)
    y_reduced, V_y = _svd_reduction_with_basis(y, accept_rate)
    a, b, diag = cca(x_reduced, y_reduced, backend)
    return x_reduced, y_reduced, a, b, diag, V_k, V_y


def compute_affine_mapping(early_acts: Tensor, late_acts: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Finds the optimal Affine Map (W, b) using Ridge Regression.
    Allows Rotation + Scaling + Shearing.
    """
    # 1. Force Float32 for precision
    X = early_acts.float()
    Y = late_acts.float()
    
    # 2. Center the data (Solving for Bias implicitly)
    mu_x = X.mean(dim=0, keepdim=True)
    mu_y = Y.mean(dim=0, keepdim=True)
    
    X_centered = X - mu_x
    Y_centered = Y - mu_y

    # 3. Solve W = (X^T X + alpha*I)^-1 @ X^T Y
    # We use Cholesky solve or standard solve for stability
    
    # Covariance matrices
    # If N (tokens) < D (hidden dim), this is rank deficient, so alpha is required.
    Cov_XX = X_centered.T @ X_centered
    Cov_XY = X_centered.T @ Y_centered
    
    # Add Ridge Regularization (Tikhonov) to diagonal
    eye = torch.eye(Cov_XX.shape[0], device=X.device, dtype=X.dtype)
    Cov_XX_reg = Cov_XX + (alpha * eye)
    
    # Solve
    # W has shape [Dim, Dim]
    W = torch.linalg.solve(Cov_XX_reg, Cov_XY)
    
    # 4. Apply transformation
    # Y_pred = (X - mu_x) @ W + mu_y
    transformed = (X_centered @ W) + mu_y
    
    return transformed.to(early_acts.dtype)


def compute_orthogonal_mapping(early_acts, late_acts, max_samples: int | None = 20000):
    print("orth")
    x_full = early_acts.float()
    y_full = late_acts.float()

    num_tokens = x_full.shape[0]
    print("num_tokens", num_tokens)

    if num_tokens > max_samples:
        # Create random indices
        indices = torch.randperm(num_tokens, device=x_full.device)[:max_samples]
        x_fit = x_full[indices]
        y_fit = y_full[indices]
    else:
        x_fit = x_full
        y_fit = y_full

    # Center the data
    mu_x = x_fit.mean(0)
    mu_y = y_fit.mean(0)
    x = x_fit - mu_x
    y = y_fit - mu_y

    # Procrustes solution
    M = y.T @ x
    print("svd")
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    print("svd done")

    R = U @ Vh

    print("applying to full dataset")

    # Apply to the full dataset

    # R_user = Vh.T @ U.T
    # transformed = x @ R_user.T + mu_y
    
    # Map
    # R = Vh.T @ U.T
    # transformed = x @ R.T + mu_y

    real_mu_x = x_full.mean(0, keepdim=True)
    real_mu_y = y_full.mean(0, keepdim=True)
    
    # Transform: (X - mu_x) @ R.T + mu_y
    # Note: R.T is the inverse rotation
    print("transforming")
    transformed = (x_full - real_mu_x) @ R.T + real_mu_y
    print("transforming done")
    return transformed

def compute_cosine_similarity_per_token(
    acts1: Tensor,
    acts2: Tensor
) -> tuple[float, float]:
    """
    Compute cosine similarity per token, return mean and std.

    Args:
        acts1: [num_tokens, features]
        acts2: [num_tokens, features]

    Returns:
        (mean, std) of cosine similarities across tokens
    """
    # Find zero vectors
    zero_mask1 = (acts1.norm(dim=1) == 0)
    zero_mask2 = (acts2.norm(dim=1) == 0)
    # assert zero_mask1.sum() == 0

    non_zero_mask1 = ~zero_mask1
    non_zero_mask2 = ~zero_mask2

    # print("zero_mask1", zero_mask1.sum(), "zero_mask2", zero_mask2.sum())
    # print("non_zero_mask1", non_zero_mask1.sum(), "non_zero_mask2", non_zero_mask2.sum())
    
    
    # assert (zero_mask1 == zero_mask2).all(), \
        # f"Zero vectors at different indices: {zero_mask1.sum()} vs {zero_mask2.sum()}"
    
    # Filter out zero vectors
    valid_mask = ~zero_mask1
    acts1 = acts1[valid_mask]
    acts2 = acts2[valid_mask]

    zero_mask2 = (acts2.norm(dim=1) == 0)
    valid_mask = ~zero_mask2
    acts1 = acts1[valid_mask]
    acts2 = acts2[valid_mask]
    
    cos_sims = F.cosine_similarity(acts1, acts2, dim=1)

    return cos_sims.mean().item(), cos_sims.std().item()


def compute_svcca_mapping_for_module(
    module: str,
    gpu_id: int,
    early_activations: list[dict[str, Tensor]],
    late_activations: list[dict[str, Tensor]],
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

    # 3. Apply Orthogonal Mapping (Replacing SVCCA here)
    # transformed_early = compute_orthogonal_mapping(early_acts, late_acts)
    transformed_early = compute_affine_mapping(early_acts, late_acts)
    print("orthogonal mapping done")
    print(transformed_early.shape)

    # Compute SVCCA transformation
    # accept_rate = 0.999
    # x_reduced, y_reduced, a, b, diag, V_k, V_y = svcca_transform(
    #     early_acts, late_acts, accept_rate, "svd"
    # )
    # Transform early activations using SVCCA mapping
    # Project to CCA space
    # early_in_cca = x_reduced @ a
    # late_in_cca = y_reduced @ b
    # Project back to late's original space
    # transformed_early = early_in_cca @ torch.linalg.pinv(b) @ V_y.T
    # print(early_in_cca.shape, b.shape, y_reduced.shape, "early in cca, b, y reduced shapes")
    # print(transformed_early.shape, late_acts.shape, "transformed early and late shapes")

    # Compute cosine similarity after transformation
    cos_trans_mean, cos_trans_std = compute_cosine_similarity_per_token(
        transformed_early, late_acts
    )

    # SVCCA similarity score
    # div = min(a.size(1), b.size(1))
    # svcca_sim = 1.0 - (1.0 - diag.sum() / div).item()


    print("Done!")
    results = {
        "cosine_original_mean": cos_orig_mean,
        "cosine_original_std": cos_orig_std,
        "cosine_transformed_mean": cos_trans_mean,
        "cosine_transformed_std": cos_trans_std,
        "cosine_improvement": cos_trans_mean - cos_orig_mean,
        # "svcca_similarity": svcca_sim,
        # "transformation_rank": div,
    }
    print("results", results)

    logger.info(
        f"Module {module}: orig={cos_orig_mean:.4f}, trans={cos_trans_mean:.4f}, "
        f"improvement={results['cosine_improvement']:.4f}"
    )

    # Free GPU memory
    # del early_acts, late_acts, x_reduced, y_reduced, a, b
    # del early_in_cca, late_in_cca, transformed_early
    # torch.cuda.empty_cache()

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
    model = AutoModelForCausalLM.from_pretrained(late_model, torch_dtype=torch.bfloat16)
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
        
        late_acts_path = Path("analysis/results/svcca") / f"late_acts_{batch_idx}.pth"
        early_acts_path = Path("analysis/results/svcca") / f"early_acts_{batch_idx}.pth"
        late_acts_path.parent.mkdir(parents=True, exist_ok=True)
        early_acts_path.parent.mkdir(parents=True, exist_ok=True)
        if (late_acts_path.exists() and early_acts_path.exists()):
            logger.info(f"Loading late and early activations from {late_acts_path} and {early_acts_path}")
            late_acts = torch.load(late_acts_path)
            early_acts = torch.load(early_acts_path)
        else:
            # Load early checkpoint
            logger.info(f"Loading early checkpoint: {early_model}@{early_revision}")
            early_model_obj = AutoModelForCausalLM.from_pretrained(
                early_model, revision=early_revision, device_map="auto", torch_dtype=torch.bfloat16
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
                late_model, device_map="auto", torch_dtype=torch.bfloat16
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
            torch.save( 
                late_acts,
                late_acts_path
            )
            torch.save(
                early_acts,
                early_acts_path
            )
            del late_model_obj
            torch.cuda.empty_cache()

        first_module = list(early_acts[0].keys())[0]
        print("early_acts", early_acts[0][first_module].shape)
        print("late_acts", late_acts[0][first_module].shape)
        print("len early_acts", len(early_acts))
        print("len late_acts", len(late_acts))
        # exit()

        for i in range(num_gpus):
            try:
                # Force initialization of cuSOLVER on each GPU sequentially
                d = f"cuda:{i}"
                dummy = torch.eye(2, device=d)
                # This dummy call initializes the lazy wrapper safely
                torch.linalg.solve(dummy, dummy) 
                torch.cuda.synchronize(d)
            except Exception as e:
                print(f"Warmup on {d} failed (might not be used): {e}")

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
