#!/usr/bin/env python3
"""
Downloads two models from Hugging Face and performs per-layer SVCCA analysis.
"""
from typing import Any
import re
from argparse import ArgumentParser
from pathlib import Path
import math

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from analysis.utils import assert_type
from analysis.svcca.distance import svcca_distance
from analysis.svcca.hooks import collect_output_activations

logger = logging.getLogger(__name__)  # not root
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def module_group_map(module_name: str) -> str:
    """Map module name to group name."""
    if "mlp" in module_name:
        return "mlp"
    elif (
        "attn" in module_name
        or "attention" in module_name
        and "ln" not in module_name
        and "layernorm" not in module_name
    ):
        return "attention"
    elif "input_layernorm" in module_name:
        return "input_ln"
    elif "layernorm" in module_name:
        return "post_ln"  # e.g. post_attention_layernorm
    else:
        print(module_name, "is other")
        return "other"


def plot_similarities(
    module_info: dict[str, dict[str, Any]],
    save_path: Path,
):
    group_sims = defaultdict(list)
    for name, info in module_info.items():
        module_group = module_group_map(name)
        group_sims[module_group].append((info["layer"], info["sim"]))

    styles = {
        "input_ln": ("solid", "blue", "Input Layer Norm"),
        "attention": ("dotted", "green", "Attention"),
        "mlp": ("dashdot", "red", "MLP"),
        "post_ln": ("dashed", "orange", "Post Layer Norm"),
    }

    plt.figure(figsize=(12, 8))

    for module_group, (linestyle, color, label) in styles.items():
        if not module_group in group_sims or not group_sims[module_group]:
            continue

        # regroup sims by layer
        layers = [layer for layer, _ in group_sims[module_group]]
        sims = [layer_sims for _, layer_sims in group_sims[module_group]]
        layer_to_sims = defaultdict(list)
        for layer, sim in zip(layers, sims):
            layer_to_sims[layer].append(sim)

        # sorted unique layer indices
        unique_layers = sorted(layer_to_sims.keys())

        # compute mean and std error
        means = []
        std_errs = []
        for layer in unique_layers:
            values = layer_to_sims[layer]
            means.append(np.mean(values))

            # std error only if multiple samples
            if len(values) > 1:
                std_errs.append(np.std(values) / np.sqrt(len(values)))
            else:
                std_errs.append(0)  # keep array aligned

        plt.plot(
            unique_layers,
            means,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        if any(e > 0 for e in std_errs):
            lo = [m - e for m, e in zip(means, std_errs)]
            hi = [m + e for m, e in zip(means, std_errs)]
            plt.fill_between(unique_layers, lo, hi, alpha=0.3, color=color)

    plt.xlabel("Layer Index")
    plt.ylabel("SVCCA Similarity")
    plt.title("Per-Layer SVCCA Similarity (Grouped)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Plot saved to: {save_path}")
    print(f"Plot saved to: {save_path}")


def extract_layer_idx(name: str) -> int | None:
    pattern = re.compile(r"\.(?:layers|h)\.(\d+)")
    matches = pattern.search(name)
    if matches:
        return int(matches.group(1))

    if "layers" in name:
        return int(name.split(".")[1])

    return None


def load_and_tokenize(dataset_name: str, subset: str, num_items: int, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "cais/wmdp" or dataset_name == "cais/mmlu":
        if subset:
            dataset = load_dataset(dataset_name, subset, split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")

        dataset = assert_type(Dataset, dataset)
        if len(dataset) > num_items:
            dataset = dataset.select(range(num_items))

        def map_mcqa(x):
            choices = [f"{i}. {choice}" for i, choice in enumerate(x["choices"])]
            prompt = " \n ".join(
                [x["question"]]
                + ["Choices: "]
                + choices
                + ["Answer: "]
                + [f"{x['answer']}"]
            )
            return {
                "input_ids": tokenizer.encode(
                    prompt, max_length=16384, truncation=True
                ),
            }

        dataset = dataset.map(map_mcqa)
    else:
        dataset = load_dataset(dataset_name, split=f"train[:{num_items}]")
        dataset = assert_type(Dataset, dataset)
        dataset = dataset.map(lambda x: {"input_ids": tokenizer.encode(x["text"])})

    # Sort by length to minimize padding
    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
    dataset = dataset.sort("length")

    # Add attention mask (all 1s initially, will be updated with padding)
    dataset = dataset.map(lambda x: {"attention_mask": [1] * len(x["input_ids"])})

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset


@torch.inference_mode()
def collect_module_activations(
    model: PreTrainedModel,
    model_name: str,
    target_modules: list[str],
    dataset: Dataset,
    batch_size: int = 1,
    tokens_per_sequence: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Collect module activations from a model.
    Args:
        model: The model to collect activations from.
        model_name: The name of the model.
        target_modules: The modules to collect activations from.
        dataset: The dataset to collect activations from.
        batch_size: The batch size to use for collecting activations.
        tokens_per_sequence: The number of tokens to sample per sequence.
            If None, all tokens will be collected. If not None, tokens will be
            sampled from the end of the sequence.
    """
    def collate_fn(batch):
        """Pad sequences to the same length within a batch."""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # Find max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_len - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.zeros(padding_len, dtype=ids.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(padding_len, dtype=mask.dtype)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # type: ignore

    module_activations = []
    for batch in tqdm(dl, desc=f"Collecting activations for {model_name}"):
        with collect_output_activations(
            model, hookpoints=target_modules
        ) as activations:
            model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )

            if tokens_per_sequence is not None:
                sample = {
                    name: activations[name][:, -tokens_per_sequence:] for name in target_modules
                }
            else:
                sample = {
                    name: activations[name] for name in target_modules
                }

            module_activations.append(sample)

    return module_activations


def compute_svcca_for_module(
    module: str,
    gpu_id: int,
    collected_activations_0: list[dict[str, torch.Tensor]],
    collected_activations_1: list[dict[str, torch.Tensor]],
) -> tuple[str, float]:
    """Compute SVCCA for a single module on a specific GPU."""
    device = f"cuda:{gpu_id}"

    first_model_acts = torch.cat(
        [
            item[module].to(device, non_blocking=True).flatten(0, 1)
            for item in collected_activations_0
        ],
        dim=0,
    ).to(torch.float32)
    second_model_acts = torch.cat(
        [
            item[module].to(device, non_blocking=True).flatten(0, 1)
            for item in collected_activations_1
        ],
        dim=0,
    ).to(torch.float32)

    similarity = (
        1.0
        - svcca_distance(
            first_model_acts,
            second_model_acts,
            0.99,
            "svd",
        ).item()
    )

    # Free GPU memory
    del first_model_acts, second_model_acts
    torch.cuda.empty_cache()

    return module, similarity


def get_module_info(
    model_names: tuple[str, str],
    dataset: Dataset,
    batch_size: int,
    module_batch_size: int = 128,
    target_layers: list[int] | None = None,
    num_gpus: int = 8,
    tokens_per_sequence: int = 1,
):
    # Modulate CPU RAM usage by batching module similarities
    print(f"module_batch_size: {module_batch_size}")
    print(f"num_gpus: {num_gpus}")

    named_modules = AutoModelForCausalLM.from_pretrained(
        model_names[0]
    ).base_model.named_modules()

    target_module_info = {}
    for name, module in named_modules:
        if (
            isinstance(module, nn.Linear)  # or
            # isinstance(module, nn.Embedding)
            # or "layernorm" in name
            # or "ln" in name
        ):
            layer = extract_layer_idx(name)
            if layer is None:
                # logger.debug(
                #     f"Skipping {name} because it doesn't match the layer pattern"
                # )
                continue
            if target_layers is not None and str(layer) not in target_layers:
                # logger.debug(
                #     f"Skipping {name} because {layer} is not in the target layers {target_layers}"
                # )
                continue

            group = module_group_map(name)
            if group == "other":
                # logger.debug(f"Skipping {name} because it's in the 'other' group")
                continue

            if target_layers is not None:
                if any(
                    info.get("group") == group and info.get("layer") == layer
                    for info in target_module_info.values()
                ):
                    # logger.debug(
                    #     f"Skipping {name} because {group} {layer} is already in the target module info"
                    # )
                    continue

            target_module_info[name] = {
                "layer": layer,
                "group": group,
            }

    logger.debug(f"Grouped target modules: {list(target_module_info.keys())}")
    logging.info("Collecting output activations...")

    module_batches = [
        list(target_module_info.keys())[i : i + module_batch_size]
        for i in range(0, len(target_module_info), module_batch_size)
    ]
    for modules_batch in module_batches:
        collected_activations = []

        for model_name in model_names:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            collected_activations.append(
                collect_module_activations(
                    model, model_name, modules_batch, dataset, batch_size=batch_size, tokens_per_sequence=tokens_per_sequence
                )
            )

        # Run SVCCA computation in parallel across GPUs using ThreadPoolExecutor
        print(f"Parallelizing SVCCA across {num_gpus} GPUs...")

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # Submit tasks with round-robin GPU assignment
            futures = {}
            for idx, module in enumerate(modules_batch):
                gpu_id = idx % num_gpus
                future = executor.submit(
                    compute_svcca_for_module,
                    module,
                    gpu_id,
                    collected_activations[0],
                    collected_activations[1],
                )
                futures[future] = module

            # Collect results with progress bar
            with tqdm(total=len(modules_batch), desc="Computing SVCCA") as pbar:
                for future in futures:
                    module, similarity = future.result()
                    target_module_info[module]["sim"] = similarity
                    logging.info(
                        f"SVCCA {(target_module_info[module]['layer'], target_module_info[module]['group'], module)}: "
                        f"{similarity}"
                    )
                    pbar.update(1)

    return target_module_info


def plot(args, data_path: Path, output_path: Path):

    module_info = torch.load(data_path)

    plot_similarities(
        module_info,
        output_path,
    )


@torch.inference_mode()
def main(args):
    logger.info(f"Using the first model to tokenize the dataset: {args.models[0]}")
    output_path = Path("analysis/results/svcca")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_and_tokenize(
        args.dataset_name,
        args.subset,
        num_items=args.num_items,
        model_name=args.models[0],
    )

    # Compute tokens per sequence based on actual dataset length
    tokens_per_sequence = max(1, math.ceil(args.num_samples / len(dataset)))
    logger.info(f"Dataset length: {len(dataset)}, tokens per sequence: {tokens_per_sequence}")

    module_info = get_module_info(
        tuple(args.models),
        dataset,
        batch_size=args.batch_size,
        target_layers=args.target_layers,
        num_gpus=args.num_gpus,
        tokens_per_sequence=tokens_per_sequence,
    )

    run_name = (
        f"layer_sims_{args.dataset_name.split('/')[-1]}"
        f"_N={args.num_items}_n_layers={len(args.target_layers)}_"
        f"{args.models[0].split('/')[-1]}_{args.models[1].split('/')[-1]}"
    )
    file_path = output_path / run_name / "module_info_t.pth"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module_info, file_path)

    plot(args, file_path, output_path / f"{run_name}_t.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cais/wmdp")
    parser.add_argument("--subset", type=str, default="wmdp-bio")
    parser.add_argument("--target_layers", nargs="+", default=None)
    parser.add_argument("--num_items", type=int, default=10_000, help="Number of sequences/items to load from dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10_000,
        help=(
            "Target number of token activation samples for SVCCA. "
            "A fixed number of activations will be taken from tokens "
            "at the end of each sequence."
        )
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "EleutherAI/deep-ignorance-unfiltered",
            "EleutherAI/deep-ignorance-e2e-strong-filter",
            # "EleutherAI/deep-ignorance-e2e-weak-filter",
            # "EleutherAI/deep-ignorance-unfiltered-cb-lat",
            # "EleutherAI/deep-ignorance-unfiltered-cb"
        ],
    )
    args = parser.parse_args()

    assert len(args.models) == 2, "Compare two models at a time"

    main(args)
