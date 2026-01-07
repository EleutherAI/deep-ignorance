#!/usr/bin/env python3
"""
Analyze representation rotation between early and late model checkpoints using SVCCA.

This script:
1. Collects activations from early and late checkpoints on the same data points
2. Computes SVCCA module maps to align the representations
3. Computes cosine similarities before and after transformation
4. Presents results showing how much representations rotate during training
"""

import math
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
import logging
import json

from analysis.svcca.checkpoint_mapping import analyze_checkpoint_mapping, compute_cosine_similarity_per_token
from analysis.svcca.svcca import load_and_tokenize, module_group_map
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model checkpoints to compare
LATE_CHECKPOINT_MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
# LATE_CHECKPOINT_MODEL_NAME = "EleutherAI/pythia-14m"
EARLY_CHECKPOINT_MODEL_NAME = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
# EARLY_CHECKPOINT_MODEL_NAME = "EleutherAI/pythia-14m"
EARLY_CHECKPOINT = "global_step38144"
# EARLY_CHECKPOINT = "main"

checkpoints = [
    "global_step5960",
    "global_step10728",
    "global_step20264",
    # "global_step35760",
    "global_step46488",
    # "global_step50064",
    # "global_step79864"
    "global_step100128",
    # "global_step109664",
    # "global_step118008"
]


def plot_cosine_similarities(
    module_info: dict[str, dict],
    output_path: Path,
):
    """Plot cosine similarities before and after SVCCA transformation."""
    group_data = defaultdict(lambda: {"layers": [], "orig": [], "trans": [], "improvement": []})
    
    for name, info in module_info.items():
        module_group = module_group_map(name)
        if module_group == "other":
            continue
        
        layer = info["layer"]
        group_data[module_group]["layers"].append(layer)
        group_data[module_group]["orig"].append(info["cosine_original_mean"])
        group_data[module_group]["trans"].append(info["cosine_transformed_mean"])
        group_data[module_group]["improvement"].append(info["cosine_improvement"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Representation Rotation Analysis: Early vs Late Checkpoint", fontsize=16, fontweight="bold")
    
    styles = {
        "input_ln": ("solid", "blue", "Input Layer Norm"),
        "attention": ("dotted", "green", "Attention"),
        "mlp": ("dashdot", "red", "MLP"),
        "post_ln": ("dashed", "orange", "Post Layer Norm"),
    }
    
    # Plot 1: Original cosine similarity
    ax1 = axes[0, 0]
    for module_group, (linestyle, color, label) in styles.items():
        if module_group not in group_data or not group_data[module_group]["layers"]:
            continue
        
        # Sort by layer
        sorted_data = sorted(zip(
            group_data[module_group]["layers"],
            group_data[module_group]["orig"]
        ))
        layers, sims = zip(*sorted_data) if sorted_data else ([], [])
        
        if layers:
            ax1.plot(layers, sims, linestyle=linestyle, color=color, label=label,
                    linewidth=2, marker="o", markersize=4)
    
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Original Cosine Similarity (Before Transformation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Transformed cosine similarity
    ax2 = axes[0, 1]
    for module_group, (linestyle, color, label) in styles.items():
        if module_group not in group_data or not group_data[module_group]["layers"]:
            continue
        
        sorted_data = sorted(zip(
            group_data[module_group]["layers"],
            group_data[module_group]["trans"]
        ))
        layers, sims = zip(*sorted_data) if sorted_data else ([], [])
        
        if layers:
            ax2.plot(layers, sims, linestyle=linestyle, color=color, label=label,
                    linewidth=2, marker="o", markersize=4)
    
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Transformed Cosine Similarity (After SVCCA Mapping)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Improvement (transformed - original)
    ax3 = axes[1, 0]
    for module_group, (linestyle, color, label) in styles.items():
        if module_group not in group_data or not group_data[module_group]["layers"]:
            continue
        
        sorted_data = sorted(zip(
            group_data[module_group]["layers"],
            group_data[module_group]["improvement"]
        ))
        layers, improvements = zip(*sorted_data) if sorted_data else ([], [])
        
        if layers:
            ax3.plot(layers, improvements, linestyle=linestyle, color=color, label=label,
                    linewidth=2, marker="o", markersize=4)
    
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Cosine Similarity Improvement")
    ax3.set_title("Improvement from SVCCA Transformation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    
    # # Plot 4: SVCCA similarity scores
    # ax4 = axes[1, 1]
    # group_svcca = defaultdict(lambda: {"layers": [], "svcca": []})
    # for name, info in module_info.items():
    #     module_group = module_group_map(name)
    #     if module_group == "other":
    #         continue
    #     group_svcca[module_group]["layers"].append(info["layer"])
    #     # group_svcca[module_group]["svcca"].append(info["svcca_similarity"])
    
    # for module_group, (linestyle, color, label) in styles.items():
    #     if module_group not in group_svcca or not group_svcca[module_group]["layers"]:
    #         continue
        
    #     sorted_data = sorted(zip(
    #         group_svcca[module_group]["layers"],
    #         group_svcca[module_group]["svcca"]
    #     ))
    #     layers, svcca_sims = zip(*sorted_data) if sorted_data else ([], [])
        
    #     if layers:
    #         ax4.plot(layers, svcca_sims, linestyle=linestyle, color=color, label=label,
    #                 linewidth=2, marker="o", markersize=4)
    
    # ax4.set_xlabel("Layer Index")
    # ax4.set_ylabel("SVCCA Similarity")
    # ax4.set_title("SVCCA Similarity Score")
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)
    # ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def print_summary_statistics(module_info: dict[str, dict]):
    """Print summary statistics about representation rotation."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: Representation Rotation Analysis")
    print("=" * 80)
    
    # Group by module type
    by_group = defaultdict(list)
    for name, info in module_info.items():
        group = module_group_map(name)
        if group != "other":
            by_group[group].append(info)
    
    for group, infos in sorted(by_group.items()):
        print(f"\n{group.upper()} Modules:")
        print("-" * 80)
        
        orig_sims = [info["cosine_original_mean"] for info in infos]
        trans_sims = [info["cosine_transformed_mean"] for info in infos]
        improvements = [info["cosine_improvement"] for info in infos]
        # svcca_sims = [info["svcca_similarity"] for info in infos]
        
        print(f"  Original Cosine Similarity:")
        print(f"    Mean: {np.mean(orig_sims):.4f} ± {np.std(orig_sims):.4f}")
        print(f"    Min: {np.min(orig_sims):.4f}, Max: {np.max(orig_sims):.4f}")
        
        print(f"  Transformed Cosine Similarity:")
        print(f"    Mean: {np.mean(trans_sims):.4f} ± {np.std(trans_sims):.4f}")
        print(f"    Min: {np.min(trans_sims):.4f}, Max: {np.max(trans_sims):.4f}")
        
        print(f"  Improvement from SVCCA:")
        print(f"    Mean: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
        print(f"    Min: {np.min(improvements):.4f}, Max: {np.max(improvements):.4f}")
        
        # print(f"  SVCCA Similarity:")
        # print(f"    Mean: {np.mean(svcca_sims):.4f} ± {np.std(svcca_sims):.4f}")
        # print(f"    Min: {np.min(svcca_sims):.4f}, Max: {np.max(svcca_sims):.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
    - Original Cosine Similarity: How similar are the raw activations?
      (Lower = more rotation in raw space)
    
    - Transformed Cosine Similarity: How similar after SVCCA alignment?
      (Higher = better alignment, less intrinsic change)
    
    - Improvement: How much does SVCCA transformation help?
      (Positive = rotation was a significant component of change)
    
    - SVCCA Similarity: Overall representation similarity after alignment
      (Higher = more similar representations)
    """)


def save_results(module_info: dict[str, dict], output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    json_results = {}
    for name, info in module_info.items():
        json_results[name] = {
            "layer": info["layer"],
            "group": info["group"],
            "cosine_original_mean": info["cosine_original_mean"],
            "cosine_original_std": info["cosine_original_std"],
            "cosine_transformed_mean": info["cosine_transformed_mean"],
            "cosine_transformed_std": info["cosine_transformed_std"],
            "cosine_improvement": info["cosine_improvement"],
            # "svcca_similarity": info["svcca_similarity"],
            # "transformation_rank": info["transformation_rank"],
        }
    
    json_path = output_dir / "checkpoint_rotation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    
    # Save PyTorch format for compatibility
    torch_path = output_dir / "checkpoint_rotation_results.pth"
    torch.save(module_info, torch_path)
    logger.info(f"Results saved to: {torch_path}")


def load_and_tokenize_local_dataset(
    dataset_path: str,
    num_items: int,
    model_name: str,
) -> Dataset:
    """Load and tokenize a local on-disk dataset."""
    logger.info(f"Loading dataset from disk: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # If it's a DatasetDict, get the train split
    if hasattr(dataset, 'keys'):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            dataset = dataset[list(dataset.keys())[0]]
    
    # Limit number of items
    if len(dataset) > num_items:
        dataset = dataset.select(range(num_items))
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check what columns the dataset has
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # Tokenize based on available columns
    if 'text' in dataset.column_names:
        def tokenize_fn(x):
            return {"input_ids": tokenizer.encode(x["text"], max_length=16384, truncation=True)}
    elif 'input_ids' in dataset.column_names:
        # Already tokenized
        def tokenize_fn(x):
            return {"input_ids": x["input_ids"]}
    else:
        # Try to find a text-like column
        text_col = None
        for col in dataset.column_names:
            if 'text' in col.lower() or 'content' in col.lower():
                text_col = col
                break
        if text_col is None:
            raise ValueError(f"Could not find text column in dataset. Available: {dataset.column_names}")
        
        def tokenize_fn(x):
            return {"input_ids": tokenizer.encode(str(x[text_col]), max_length=16384, truncation=True)}
    
    dataset = dataset.map(tokenize_fn)
    
    # Sort by length to minimize padding
    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
    dataset = dataset.sort("length")
    
    # Add attention mask
    dataset = dataset.map(lambda x: {"attention_mask": [1] * len(x["input_ids"])})
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return dataset




def main():
    """Main function to run the checkpoint rotation analysis."""
    # Configuration
    dataset_path = "rmu_training_data/bio-forget-corpus"
    num_items = 100  # Number of examples to use (reduced for testing)
    num_samples = 2048  # Target number of token activations (reduced for testing)
    batch_size = 4  # Reduced batch size to avoid memory issues
    max_modules = 8  # Limit to 8 modules for testing
    num_gpus = 8
    sample_strategy = "space_evenly"
    
    # Output directory
    output_dir = Path("analysis/results/checkpoint_rotation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Checkpoint Rotation Analysis")
    logger.info("=" * 80)
    logger.info(f"Early checkpoint: {EARLY_CHECKPOINT_MODEL_NAME}@{EARLY_CHECKPOINT}")
    logger.info(f"Late checkpoint: {LATE_CHECKPOINT_MODEL_NAME}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Number of examples: {num_items}")
    logger.info(f"Max modules (testing): {max_modules}")
    logger.info("=" * 80)
    
    # Load and tokenize dataset
    logger.info("Loading and tokenizing dataset...")
    dataset = load_and_tokenize_local_dataset(
        dataset_path,
        num_items,
        LATE_CHECKPOINT_MODEL_NAME,  # Use late model for tokenization
    )
    
    # Compute tokens per sequence
    tokens_per_sequence = max(1, math.ceil(num_samples / len(dataset)))
    logger.info(f"Dataset length: {len(dataset)}, tokens per sequence: {tokens_per_sequence}")
    
    # Run analysis
    logger.info("Starting checkpoint mapping analysis...")
    module_info = analyze_checkpoint_mapping(
        early_model=EARLY_CHECKPOINT_MODEL_NAME,
        early_revision=EARLY_CHECKPOINT,
        late_model=LATE_CHECKPOINT_MODEL_NAME,
        dataset=dataset,
        batch_size=batch_size,
        module_batch_size=max_modules,  # Process all modules in one batch
        target_layers=None,
        num_gpus=num_gpus,
        tokens_per_sequence=tokens_per_sequence,
        sample_strategy=sample_strategy,
        max_modules=max_modules,  # Limit total modules
    )
    
    # Save results
    logger.info("Saving results...")
    save_results(module_info, output_dir)
    
    # Print summary
    print_summary_statistics(module_info)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_path = output_dir / "checkpoint_rotation_analysis.png"
    plot_cosine_similarities(module_info, plot_path)
    
    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

