from pathlib import Path
import json
from argparse import ArgumentParser
import gc
import ast

import torch
import pandas as pd
from huggingface_hub import list_models
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

from analysis.utils import assert_type


# Configuration
ORG_NAME = "open-unlearning"
TASK_TAG = "tofu"
BASE_MODEL_TAG = "Llama-3.2-1B-Instruct" 
FORGET_SPLIT_TAG = "forget10"

ALGO_ALIASES = {
    "AltPO": ["AltPO"],
    "NPO": ["NPO"],
    "SimNPO": ["SimNPO"],
    "GradDiff": ["GradDiff", "GradientDifference"],
    # "GA": ["GradAscent", "GradientAscent", "GA"], # GA often listed as GradAscent
    # "WGA": ["WGA", "WeightedGA"],
    "IdkPO": ["IdkDPO", "IdkNLL", "IdkPO", "IDK"], # Repo uses IdkDPO or IdkNLL
    "RMU": ["RMU"],
    "UNDIAL": ["UNDIAL"],
}

def get_best_model_name(algo_aliases, all_models, filter_pert=False):
    """
    Filters models to find one matching the task, base model, split, and one of the algo aliases.
    Returns the first match found.
    """
    # Pre-filter for task, model, and split to reduce search space
    relevant_models = [
        m.modelId for m in all_models 
        if TASK_TAG in m.modelId 
        and BASE_MODEL_TAG in m.modelId 
        and FORGET_SPLIT_TAG in m.modelId
    ]

    # bio = biographical
    # para = paraphased
    # pert = perturbed
    
    # Select models that match aliases
    matching_models = []
    for alias in algo_aliases:
        for model_id in relevant_models:
            if alias in model_id:
                matching_models.append(model_id)
    
    # if "neg" in algo_aliases:
        # print("found neg", matching_models)

    # if filter_pert:
        # matching_models = [model for model in matching_models if "pert" in model]

    if matching_models:
        # If any models have "epoch" in them, select the one with the highest epoch
        # Formatted like epoch10, epoch15
        epoch_models = [model for model in matching_models if "epoch" in model]
        if epoch_models:
            epoch_models.sort(key=lambda x: int(x.split("epoch")[1]))
            return epoch_models[-1]
        
        # Otherwise, select the first model
        return matching_models[0]

    return None

import torch

def analyze(model, tokenizer, dataset, batch_size=4):
    """
    Computes:
    1. Average Cross Entropy Loss.
    2. Average L2 Norm of hidden states per layer.
    """
    model.eval()
    
    total_loss = 0
    count_samples = 0
    
    # Variables for hidden state tracking
    total_hidden_norms = None
    total_valid_tokens = 0
    
    device = model.device
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Format: "Question Answer" concatenation
            texts = [f"{q} {a}" for q, a in zip(batch['question'], batch['answer'])]
            
            # Create inputs
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels = inputs.input_ids.clone()
            
            # Mask padding tokens so they don't contribute to loss
            labels[inputs.attention_mask == 0] = -100
            
            # Forward pass with output_hidden_states=True
            outputs = model(**inputs, labels=labels, output_hidden_states=True)
            
            # --- 1. Loss Calculation ---
            loss = outputs.loss.item()
            current_batch_size = inputs.input_ids.size(0)
            total_loss += loss * current_batch_size
            count_samples += current_batch_size
            
            # --- 2. Hidden State Norm Calculation ---
            # hidden_states is a tuple of tensors (one for embeddings + one for each layer)
            hidden_states = outputs.hidden_states
            
            # Initialize the running sum list if this is the first batch
            if total_hidden_norms is None:
                total_hidden_norms = [0.0] * len(hidden_states)
            
            # Get the attention mask to ignore padding tokens in norm calculation
            # mask shape: [batch_size, seq_len]
            mask = inputs.attention_mask
            valid_tokens_in_batch = mask.sum().item()
            total_valid_tokens += valid_tokens_in_batch
            
            for layer_idx, layer_tensor in enumerate(hidden_states):
                # layer_tensor shape: [batch_size, seq_len, hidden_dim]
                
                # Calculate L2 norm across the hidden dimension -> [batch_size, seq_len]
                token_norms = layer_tensor.norm(p=2, dim=-1)
                
                # Apply mask: Zero out norms corresponding to padding tokens
                masked_norms = token_norms * mask
                
                # Sum norms of all valid tokens in this batch
                batch_layer_sum = masked_norms.sum().item()
                
                # Accumulate
                total_hidden_norms[layer_idx] += batch_layer_sum

    # Calculate averages
    avg_loss = total_loss / count_samples if count_samples > 0 else float('nan')
    
    avg_hidden_norms = []
    if total_valid_tokens > 0 and total_hidden_norms is not None:
        avg_hidden_norms = [s / total_valid_tokens for s in total_hidden_norms]
            
    return {
        "loss": avg_loss,
        "hidden_state_norms": avg_hidden_norms
    }


def get_model_names():
    all_models = list(list_models(author=ORG_NAME, limit=2000))
    
    print("Finding models for each unlearning algorithm...")

    # Find a model for each unlearning algorithm
    unlearning_models = {}
    for algo_name, aliases in ALGO_ALIASES.items():
        model_name = get_best_model_name(aliases, all_models)

        if not model_name:
            print(f"  [Warning] Could not find a model for {algo_name}. Skipping.")
            continue
        else:
            print(f"  Found {algo_name} model: {model_name}")

        unlearning_models[algo_name] = model_name

    # Find the global Positive and Negative control models
    control_targets = {
        "positive": ["positive", "pos"],
        "negative": ["negative", "neg"]
    }

    print("Finding models for positive and negative control models...")

    for key, aliases in control_targets.items():
        model_name = get_best_model_name(aliases, all_models, filter_pert=True)
        
        if model_name:
            unlearning_models[key] = model_name
            print(f"  Found {key} model: {model_name}")
        else:
            print(f"  [Warning] Could not find global '{key}' model.")

    return unlearning_models


def collect_model_data(data: Dataset, model_name: str, const_data: dict, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device) # type: ignore
        
    analysis = analyze(model, tokenizer, data)

    gc.collect()

    return {
        "Model Name": model_name,
        "Average Test Loss": analysis["loss"],
        **const_data,
        "Hidden State Norms": analysis["hidden_state_norms"],
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="tofu_unlearning_models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    output_file = Path(f"analysis/results/unlearning/{args.run_name}.csv")
    model_ids_file = Path(f"analysis/results/unlearning/{args.run_name}.json")

    # Load data
    dataset_name = "locuslab/tofu"
    data = load_dataset(dataset_name, FORGET_SPLIT_TAG, split="train")
    data = assert_type(Dataset, data)

    const_data = {
        "Dataset": dataset_name,
        "Split": FORGET_SPLIT_TAG,
        "Base Model": BASE_MODEL_TAG,
    }

    # Resolve model names
    if not model_ids_file.exists() or args.overwrite:
        unlearning_models = get_model_names()            
        with open(model_ids_file, "w") as f:
            json.dump(unlearning_models, f, indent=4)
    else:
        print(f"Loading model IDs from {model_ids_file}")
        with open(model_ids_file, "r") as f:
            unlearning_models = json.load(f)

    # Collect data
    if not output_file.exists() or args.overwrite:
        results = []
        for algorithm, model_name in unlearning_models.items():
            const_data["Algorithm"] = algorithm
            model_data = collect_model_data(data, model_name, const_data, device)
            results.append(model_data)

        df = pd.DataFrame(results)    
        df.to_csv(output_file, index=False)
    else:   
        print(f"Loading results from {output_file}")
        df = pd.read_csv(output_file)


    # Analyze results
    print(f"\nProcessing complete. Results saved to {output_file}")
    print(df[["Algorithm", "Average Test Loss", "Hidden State Norms"]])


    # Produce a line plot of norms across layers with a line for each algorithm
    plt.figure(figsize=(10, 6))
    for algorithm in df["Algorithm"].unique():
        if algorithm == "positive" or algorithm == "negative":
            print("found", algorithm, "continuing")

        df_algorithm = df[df["Algorithm"] == algorithm]
        norms = df_algorithm["Hidden State Norms"].iloc[0]
        if isinstance(norms, str):
            norms = ast.literal_eval(norms)

        plt.plot(list(range(len(norms))), norms, label=algorithm)

    # Add title: "Average Hidden State Norm Across Layers"
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Hidden State Norm")

    plt.legend()
    plt.savefig("analysis/results/unlearning/unlearning_norms.png")


if __name__ == "__main__":
    main()