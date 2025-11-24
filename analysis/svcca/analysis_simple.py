#!/usr/bin/env python3
"""
Deep Ignorance Model Analysis: SVCCA Comparison
Downloads models from Hugging Face and performs per-layer SVCCA analysis.
"""
from contextlib import contextmanager
from typing import Any
import re
from argparse import ArgumentParser
from pathlib import Path

from collections import defaultdict
from functools import partial
from torch import nn
import shelve
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

# import warnings
# warnings.filterwarnings("ignore")


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
                std_errs.append(0)   # keep array aligned

        plt.plot(
            unique_layers, means,
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


def load_and_tokenize(dataset_name: str, N: int, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset_name == "cais/wmdp":
        dataset = load_dataset(dataset_name, "wmdp-bio", split="test")
        dataset = assert_type(Dataset, dataset)

        def map_bio(x):
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

        dataset = dataset.map(map_bio)
    else:
        dataset = load_dataset(dataset_name, split=f"train[:{10_000}]")
        dataset = assert_type(Dataset, dataset)
        dataset = dataset.map(lambda x: {"input_ids": tokenizer.encode(x["text"])})

    dataset = dataset.select(range(N))
    dataset.set_format(type="torch", columns=["input_ids"])

    return dataset


@torch.inference_mode()
def collect_module_activations(
    model: PreTrainedModel,
    model_name: str,
    target_modules: list[str],
    dataset: Dataset,
    debug: bool = False,
) -> list[dict[str, torch.Tensor]]:
    dl = DataLoader(dataset, batch_size=1, shuffle=False)  # type: ignore

    module_activations = []
    for batch in tqdm(dl, desc=f"Collecting activations for {model_name}"):
        with collect_output_activations(
            model, hookpoints=target_modules
        ) as activations:
            model(batch["input_ids"].to(model.device))

            module_activations.append(
                {name: activations[name].squeeze() for name in target_modules}
            )

    return module_activations


def get_module_info(
    model_names: tuple[str, str],
    dataset: Dataset,
    device: str,
    module_batch_size: int = 128,
    debug: bool = False,
):
    # Modulate CPU RAM usage by batching module similarities
    print(f"module_batch_size: {module_batch_size}")

    named_modules = AutoModelForCausalLM.from_pretrained(
        model_names[0]
    ).base_model.named_modules()

    target_module_info = {}
    for name, module in named_modules:
        if (
            isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
            or "layernorm" in name
            or "ln" in name
        ):
            layer = extract_layer_idx(name)
            if layer is None:
                logger.debug(
                    f"Skipping {name} because it doesn't match the layer pattern"
                )
                continue

            group = module_group_map(name)
            if group == "other":
                logger.debug(f"Skipping {name} because it's in the 'other' group")
                continue

            target_module_info[name] = {
                "layer": layer,
                "group": group,
            }

    logger.debug(f"Grouped target modules: {list(target_module_info.keys())}")
    logging.info("Collecting output activations...")

    models = [
        AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        for name in model_names
    ]

    if debug:
        # Only collect first k modules
        num_modules = 10
        target_module_info = {
            k: v
            for k, v in target_module_info.items()
            if k in list(target_module_info.keys())[:num_modules]
        }

    module_batches = [
        list(target_module_info.keys())[i : i + module_batch_size]
        for i in range(0, len(target_module_info), module_batch_size)
    ]
    for modules in module_batches:
        collected_activations = []
        for model, model_name in zip(models, model_names):
            collected_activations.append(
                collect_module_activations(model, model_name, modules, dataset)
            )

        for module in tqdm(modules, desc=f"Computing SVCCA for {len(modules)} modules"):
            first_model_acts = (
                torch.cat([item[module].to(device) for item in collected_activations[0]], dim=0)
                .to(torch.float32)
            )
            second_model_acts = (
                torch.cat([item[module].to(device) for item in collected_activations[1]], dim=0)
                .to(torch.float32)
            )

            target_module_info[module]["sim"] = (
                1.0
                - svcca_distance(  # type: ignore
                    first_model_acts,
                    second_model_acts,
                    0.99,
                    "svd",
                ).item()
            )
            logging.info(
                f"SVCCA {(target_module_info[module]['layer'], name)}: "
                f"{target_module_info[module]['sim']}"
            )

    return target_module_info


@torch.inference_mode()
def main(args):
    logger.info(f"Using the first model to tokenize the dataset: {args.models[0]}")

    dataset = load_and_tokenize(args.dataset_name, N=args.N, model_name=args.models[0])

    if args.debug:
        cache = {}
        cache["module_info"] = get_module_info(
            tuple(args.models), dataset, args.device, debug=args.debug
        )
    else:
        with shelve.open(f"cache/{args.cache_name}") as cache:
            if "module_info" not in cache:
                cache["module_info"] = get_module_info(
                    tuple(args.models), dataset, args.device, debug=args.debug
                )

            module_info = cache["module_info"]

    file_name = (
        f"layer_sims_{args.dataset_name.split('/')[-1]}"
        f"{args.cache_name}{args.debug}"
        f"_N={len(dataset)}.png"
    )
    file_path = Path("analysis/results/svcca") / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    plot_similarities(
        module_info,
        file_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache_name", type=str, default="test512.3")
    parser.add_argument("--dataset_name", type=str, default="cais/wmdp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "EleutherAI/deep-ignorance-unfiltered",
            # EleutherAI/deep-ignorance-e2e-strong-filter
            "EleutherAI/deep-ignorance-e2e-weak-filter",
        ],
    )
    args = parser.parse_args()

    assert len(args.models) == 2, "Compare two models at a time"

    main(args)
