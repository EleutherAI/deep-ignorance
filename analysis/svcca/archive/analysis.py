#!/usr/bin/env python3
"""
Deep Ignorance Model Analysis: SVCCA Comparison
Downloads models from Hugging Face and performs per-layer SVCCA analysis.
"""
from contextlib import contextmanager
from typing import Any, Mapping
import re
from argparse import ArgumentParser

from collections import defaultdict
from functools import partial
from torch import nn
import pandas as pd
import shelve
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from datasets import load_dataset, Dataset
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from analysis.utils import assert_type
from analysis.distance import svcca_distance

logger = logging.getLogger(__name__)  # not root
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# import warnings
# warnings.filterwarnings("ignore")


@torch.inference_mode()
@contextmanager
def collect_activations(
    model: PreTrainedModel, hookpoints: list[str], token=-1, input_acts: bool = False
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

            if token != -1:
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
        # activations.clear()
        for handle in handles:
            handle.remove()


collect_input_activations = partial(collect_activations, input_acts=True)
collect_output_activations = partial(collect_activations, input_acts=False)


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
    similarities: Mapping[tuple[int, str], list[float]],
    save_path="layer_similarities.png",
):
    grouped = defaultdict(list)
    for (layer_idx, mtype), sims in similarities.items():
        grouped[mtype].append((layer_idx, sims))

    styles = {
        "input_ln": ("solid", "blue", "Input Layer Norm"),
        "attention": ("dotted", "green", "Attention"),
        "mlp": ("dashdot", "red", "MLP"),
        "post_ln": ("dashed", "orange", "Post Layer Norm"),
    }

    plt.figure(figsize=(12, 8))

    for mtype, (linestyle, color, label) in styles.items():
        if mtype in grouped and grouped[mtype]:
            grouped[mtype].sort(key=lambda x: x[0])
            layers = [layer for layer, _ in grouped[mtype]]
            layer_values = [values for _, values in grouped[mtype]]

            means = [np.mean(values) for values in layer_values]
            plt.plot(
                layers,
                means,
                linestyle=linestyle,
                color=color,
                label=label,
                linewidth=2,
                marker="o",
                markersize=4,
            )

            if len(layer_values[0]) > 1:
                std_errs = [
                    np.std(values) / np.sqrt(len(values)) for values in layer_values
                ]
                lo = [v - std_err for v, std_err in zip(means, std_errs)]
                hi = [v + std_err for v, std_err in zip(means, std_errs)]
                plt.fill_between(layers, lo, hi, alpha=0.3, color=color)

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


def get_weight_statistics(models, tokenizer, plot=False):

    W_Es = [model.get_input_embeddings().weight.detach() for model in models]
    W_Us = [model.get_output_embeddings().weight.detach() for model in models]

    results = []

    with shelve.open("cache/cache") as cache:
        # SVCCA metrics
        if "svcca_E" not in cache:
            cache["svcca_E"] = (
                1.0 - svcca_distance(W_Es[0], W_Es[1], 0.99, "svd").item()
            )
        if "svcca_U" not in cache:
            cache["svcca_U"] = (
                1.0 - svcca_distance(W_Us[0], W_Us[1], 0.99, "svd").item()
            )

        results.append(
            {
                "description": "SVCCA between W_E",
                "metric": "svcca_E",
                "value": cache["svcca_E"],
            }
        )
        results.append(
            {
                "description": "SVCCA between W_U",
                "metric": "svcca_U",
                "value": cache["svcca_U"],
            }
        )

        # Inner products
        if "inner_products_E" not in cache:
            cache["inner_products_E"] = (W_Es[0] @ W_Es[1].T).mean().item()
        if "inner_products_U" not in cache:
            cache["inner_products_U"] = (W_Us[0] @ W_Us[1].T).mean().item()

        results.append(
            {
                "description": "Mean inner product between W_E",
                "metric": "inner_products_E",
                "value": cache["inner_products_E"],
            }
        )
        results.append(
            {
                "description": "Mean inner product between W_U",
                "metric": "inner_products_U",
                "value": cache["inner_products_U"],
            }
        )

        # Cosine similarity metrics
        if "cosine_sims_E" not in cache:
            cache["cosine_sims_E"] = cosine_similarity(W_Es[0], W_Es[1]).mean().item()
        if "cosine_sims_U" not in cache:
            cache["cosine_sims_U"] = cosine_similarity(W_Us[0], W_Us[1]).mean().item()

        results.append(
            {
                "description": "Mean cosine sim between W_E",
                "metric": "cosine_sims_E",
                "value": cache["cosine_sims_E"],
            }
        )
        results.append(
            {
                "description": "Mean cosine sim between W_U",
                "metric": "cosine_sims_U",
                "value": cache["cosine_sims_U"],
            }
        )

    if plot:
        E_sims = cosine_similarity(W_Es[0], W_Es[1])
        U_sims = cosine_similarity(W_Us[0], W_Us[1])

        for sims, name in [(E_sims, "E_sims"), (U_sims, "U_sims")]:
            diagonal_sims = torch.diag(sims)
            diagonal_sims = diagonal_sims[diagonal_sims > 0].cpu().numpy()
            mean_sim = float(np.mean(diagonal_sims)) if diagonal_sims.size else 0.0

            plt.figure(figsize=(8, 5))
            plt.hist(diagonal_sims, bins=50)
            plt.axvline(mean_sim, linestyle="--")
            plt.title("Distribution of Cosine Similarities Between Corresponding Rows")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # fig = px.histogram(
            #     x=diagonal_sims,
            #     nbins=50,
            #     title="Distribution of Cosine Similarities Between Corresponding Rows",
            #     labels={
            #         'x': 'Cosine Similarity',
            #         'y': 'Frequency'
            #     },
            #     template='plotly_white'
            # )

            # # Customize the layout
            # fig.update_layout(
            #     xaxis_title="Cosine Similarity",
            #     yaxis_title="Count",
            #     showlegend=False,
            #     width=800,
            #     height=500
            # )

            # fig.add_vline(
            #     x=mean_sim,
            #     line_dash="dash",
            #     line_color="red",
            #     annotation_text=f"Mean: {mean_sim:.3f}"
            # )

            # fig.write_image(f"{name}.png", scale=5)

        # Select rows with cosine similarity > 0.99
        indices = torch.where(E_sims > 0.99)[0].cpu().numpy()
        # Decode the tokens
        # decoded_tokens = [tokenizer.decode(token) for token in indices]
        # logger.debug("Unchanged tokens:")
        # logger.debug(decoded_tokens)

        # U_sims
        U_sims = torch.diag(U_sims)
        U_sims = U_sims[U_sims > 0].cpu().numpy()
        # decoded_tokens = [tokenizer.decode(token) for token in indices]
        # logger.debug("Unchanged tokens:")
        # logger.debug(decoded_tokens)

    return pd.DataFrame(results)


def flatten_time(x: Tensor) -> Tensor:
    if x.dim() == 3:
        b, s, f = x.shape
        return x.reshape(b * s, f)
    if x.dim() == 2:
        return x
    return x.flatten(1)


def extract_layer_idx(name: str) -> int | None:
    pattern = re.compile(r"\.(?:layers|h)\.(\d+)")
    matches = pattern.search(name)
    if matches:
        return int(matches.group(1))

    if "layers" in name:
        return int(name.split(".")[1])

    return None


def load_and_tokenize(dataset_name: str, tokenizer, N: int):
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
def collect_module_activations(model, model_name, modules, dataset, device, debug=False):
    dl = DataLoader(dataset, batch_size=1, shuffle=False)  # type: ignore

    print("usage", torch.cuda.memory_allocated() / 1024**3, "GB")
    module_activations = []

    for batch in tqdm(dl, desc=f"Collecting activations for {model_name}"):
        with collect_output_activations(
            model, hookpoints=modules
        ) as activations:
            model(batch["input_ids"])

            for module in modules:
                # batch size of 1
                activations[module] = activations[module].cpu().squeeze()
            module_activations.append(activations)


    print(module_activations[0].keys())
    print(module_activations[0][modules[0]].shape)

    concatenated_module_activations = {
        module: torch.cat([acts[module] for acts in module_activations], dim=0) 
        for module in modules
    }

    # if debug:
    #     assert not torch.isnan(module_activations).any(), f"NaN in {module} for {model_name}"
    #     assert not torch.isinf(module_activations).any(), f"Inf in {module} for {model_name}"
    #     assert torch.isfinite(module_activations).all(), f"Non-finite in {module} for {model_name}"
    #     assert module_activations.shape[0] > 0 and module_activations.shape[1] > 0
    #     assert module_activations.dim() == 2, f"{module} shape {module_activations.shape} not [T,F]"

    return concatenated_module_activations

def compute_sims_2(model_names, dataset, model, device):
    grouped_target_modules = defaultdict(list)
    for name, layer in model.base_model.named_modules():
        if (
            isinstance(layer, nn.Embedding)
            or isinstance(layer, nn.Linear)
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

            grouped_target_modules[(layer, group)] = name

    logger.debug(f"Grouped target modules: {grouped_target_modules}")
    logging.info("Collecting output activations...")
    print("grouped_target_modules", grouped_target_modules)

    # Collect activations module by module to conserve CPU RAM
    models = [
        AutoModelForCausalLM.from_pretrained(name, device_map="auto") for name in model_names
    ]
    
    sims = defaultdict(list)
    
    groups_per_collect = 128 # Was 32
    groups = {}
    i = 0
    for (layer, group), mod_name in tqdm(grouped_target_modules.items(), desc="Grouping target modules"):
        groups[(layer, group)] = mod_name
        i += 1
        if i == groups_per_collect:
            # Collect activations
            group_module_names = [mod for mod in groups.values()]
            
            collected_activations = []
            for model, model_name in zip(models, model_names):
                print("usage1", torch.cuda.memory_allocated() / 1024**3, "GB")
                collected_activations.append(collect_module_activations(model, model_name, group_module_names, dataset, device))
                print("usage2", torch.cuda.memory_allocated() / 1024**3, "GB")

            # Process activations
            # always len = 1
            for (layer, group), module in groups.items():
                sims[(layer, module)].append(
                    1.0
                    - svcca_distance(  # type: ignore
                        collected_activations[0][module].to(torch.float32).to(device),
                        collected_activations[1][module].to(torch.float32).to(device),
                        0.99,
                        "svd",
                    ).item()
                )
                logging.info(f"SVCCA {(layer, module)}: {sims[(layer, module)]}")
            del collected_activations
            groups = {}
            i = 0
        

    return sims


# def compute_sims(model_names, dataset, model, device):
#     grouped_target_modules = defaultdict(list)
#     for name, layer in model.base_model.named_modules():
#         if (
#             isinstance(layer, nn.Embedding)
#             or isinstance(layer, nn.Linear)
#             or "layernorm" in name
#             or "ln" in name
#         ):
#             layer = extract_layer_idx(name)
#             if layer is None:
#                 logger.debug(
#                     f"Skipping {name} because it doesn't match the layer pattern"
#                 )
#                 continue

#             group = module_group_map(name)
#             if group == "other":
#                 logger.debug(f"Skipping {name} because it's in the 'other' group")
#                 continue

#             grouped_target_modules[(layer, group)].append(name)

#     logger.debug(f"Grouped target modules: {grouped_target_modules}")
#     print("Grouped target modules", grouped_target_modules)
#     logging.info("Collecting output activations...")

#     # Collect activations module by module to conserve CPU RAM
#     models = [
#         AutoModelForCausalLM.from_pretrained(name, device_map="auto") for name in model_names
#     ]
#     sims = defaultdict(list)
#     for (layer, pretty_name), module in grouped_target_modules.items():
#         # for module in modules:
#         collected_activations = []
#         for model, model_name in zip(models, model_names):
#             print("usage1", torch.cuda.memory_allocated() / 1024**3, "GB")
#             collected_activations.append(collect_module_activations(model, model_name, module, dataset, device))
#             print("usage2", torch.cuda.memory_allocated() / 1024**3, "GB")

#         # for module in modules:
#             # model_1_acts = collected_activations[0]
#             # model_2_acts = collected_activations[1][module]
#         sims[(layer, module)].append(
#             1.0
#             - svcca_distance(  # type: ignore
#                 collected_activations[0][module].to(torch.float32).to(device),
#                 collected_activations[1][module].to(torch.float32).to(device),
#                 0.99,
#                 "svd",
#             ).item()
#         )
#         del collected_activations
#         logging.info(f"SVCCA {(layer, module)}: {sims[(layer, module)]}")

#     return sims

@torch.inference_mode()
def main(args):
    models = [AutoModelForCausalLM.from_pretrained(name) for name in args.models]
    logger.info(f"Using the first model to tokenize the dataset: {args.models[0]}")
    tokenizer = AutoTokenizer.from_pretrained(args.models[0])

    if args.compute_weight_statistics:
        logging.info("Computing embedding and unembedding weight statistics...")
        statistics_df = get_weight_statistics(models, tokenizer, plot=True)
        for _, row in statistics_df.iterrows():
            logging.info(row["description"])
            logging.info(row["metric"])
            logging.info(row["value"])
            logging.info("")

    dataset = load_and_tokenize(args.dataset_name, tokenizer=tokenizer, N=args.N)

    with shelve.open(f"cache/{args.cache_name}") as cache:
        if "sims" not in cache:
            cache["sims"] = compute_sims_2(args.models, dataset, models[0], args.device)

        sims = cache["sims"]

    grouped_sims: Mapping[tuple[int, str], list[float]] = defaultdict(list)
    for (layer, module), sim in sims.items():
        # print(layer, module, module_group_map(module))
        grouped_sims[(layer, module_group_map(module))].append(sim)

    plot_similarities(
        grouped_sims,
        (
            f"layer_similarities_{args.dataset_name.split('/')[-1]}"
            f"{args.cache_name}"
            f"_N={len(dataset)}.png"
        )
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache_name", type=str, default="test512.3")

    # dataset_name = "RonenEldan/TinyStories"
    # dataset_name = "EleutherAI/deep-ignorance-annealing-mix"
    parser.add_argument("--dataset_name", type=str, default="cais/wmdp")
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--compute_weight_statistics", action="store_true")
    args = parser.parse_args()

    assert len(args.models) == 2, "Compare two models at a time"

    main(args)
