#!/usr/bin/env python3
"""
Get available revisions from HuggingFace models for checkpoint analysis.
"""

import re
from huggingface_hub import HfApi
from typing import List, Tuple
import json

from analysis.format_checkpoints import format_checkpoints

def get_model_revisions(model_name: str) -> List[str]:
    """Get all available revisions for a HuggingFace model."""
    api = HfApi()
    try:
        # Get all commits/revisions
        commits = api.list_repo_commits(model_name, repo_type="model")
        revisions = [commit.commit_id for commit in commits]

        # Also get any named revisions/branches
        refs = api.list_repo_refs(model_name, repo_type="model")

        # Add branch names
        for ref in refs.branches:
            if ref.name != "main":  # Skip main branch
                revisions.append(ref.name)

        # Add tag names
        for ref in refs.tags:
            revisions.append(ref.name)

        return revisions

    except Exception as e:
        print(f"Error fetching revisions for {model_name}: {e}")
        return []

def parse_global_step(revision: str):
    """Parse global step number from revision name."""
    # Look for patterns like "global_step100128"
    match = re.match(r'global_step(\d+)', revision)
    if match:
        return int(match.group(1))

    print(f"Warning: Could not parse global step from revision: {revision}, skipping...")
    return None

def get_checkpoint_revisions(model_name: str, const_data: dict[str, str]) -> List[Tuple[int, str]]:
    """Get checkpoint revisions sorted by step number."""
    revisions = get_model_revisions(model_name)

    checkpoints = []
    for revision in revisions:
        step = parse_global_step(revision)
        if step is not None:
            checkpoints.append({
                "step": step,
                "revision": revision,
                **const_data
            })

    # Sort by step number
    checkpoints.sort(key=lambda x: x["step"])
    return checkpoints

def save_available_checkpoints(
    hf_model_stages: dict[str, dict[str, str]],
    output_file: str,
    verbose: bool = True,
):
    """Save available checkpoints to a file."""
    all_checkpoints = {}

    for model_name, info in hf_model_stages.items():
        all_checkpoints[model_name] = get_checkpoint_revisions(
            model_name, 
            {
                "stage": info["stage"]
            }
        )

    with open(output_file, 'w') as f:
        json.dump(all_checkpoints, f, indent=2)


def main():
    hf_models = {
        "EleutherAI/deep-ignorance-pretraining-stage-unfiltered": {
            "stage": "pretraining",
        },
        "EleutherAI/deep-ignorance-unfiltered": {
            "stage": "annealing",
        },
        "EleutherAI/annealing_baseline_ga_v3_interleaved_1_in_50_ga_lr_scale-0.001_gd_lr-0.00012_gclip-0.5": {
            "stage": "annealing",
        },
    }
    checkpoints_file = f"/mnt/ssd-1/lucia/deep-ignorance/analysis/results/available_checkpoints.json"

    save_available_checkpoints(hf_models, checkpoints_file)

    print(f"Setting up metadata for deep ignorance unfiltered run...")
    hf_models = {
        "pretraining": "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        "annealing": "EleutherAI/deep-ignorance-unfiltered",
    }
    model_nickname = "deep_ignorance_unfiltered"
    output_path = f'/mnt/ssd-1/lucia/deep-ignorance/analysis/results/{model_nickname}_checkpoints.json'
    format_checkpoints(checkpoints_file, output_path, hf_models)

    print(f"Setting up metadata for annealing unlearning run...")
    hf_models = {
        "pretraining": "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        "annealing": "EleutherAI/annealing_baseline_ga_v3_interleaved_1_in_50_ga_lr_scale-0.001_gd_lr-0.00012_gclip-0.5"
    }
    model_nickname = "unlearning_annealing"
    output_path = f'/mnt/ssd-1/lucia/deep-ignorance/analysis/results/{model_nickname}_checkpoints.json'
    format_checkpoints(checkpoints_file, output_path, hf_models) 


if __name__ == "__main__":
    main()