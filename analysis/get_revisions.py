#!/usr/bin/env python3
"""
Get available revisions from HuggingFace models for checkpoint analysis.
"""

import re
from huggingface_hub import HfApi
from typing import List, Tuple
import json

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

def parse_global_step(revision: str) -> int:
    """Parse global step number from revision name."""
    # Look for patterns like "global_step100128"
    match = re.match(r'global_step(\d+)', revision)
    if match:
        return int(match.group(1))
    return None

def get_checkpoint_revisions(model_name: str) -> List[Tuple[int, str]]:
    """Get checkpoint revisions sorted by step number."""
    revisions = get_model_revisions(model_name)

    checkpoints = []
    for revision in revisions:
        step = parse_global_step(revision)
        if step is not None:
            checkpoints.append((step, revision))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    output_file = "/mnt/ssd-1/lucia/deep-ignorance/analysis/available_checkpoints.json"

    # Models to analyze
    models = {
        "pretraining": "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        "annealing": "EleutherAI/deep-ignorance-unfiltered"
    }

    all_checkpoints = {}

    for stage, model_name in models.items():
        print(f"\nFetching revisions for {model_name}...")
        checkpoints = get_checkpoint_revisions(model_name)

        print(f"Found {len(checkpoints)} checkpoint revisions:")
        for step, revision in checkpoints[:5]:  # Show first 5
            print(f"  {step}: {revision}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
            print(f"  Last: {checkpoints[-1][0]}: {checkpoints[-1][1]}")

        all_checkpoints[stage] = checkpoints

    # Save results
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for stage, checkpoints in all_checkpoints.items():
            serializable[stage] = [{"step": step, "revision": rev} for step, rev in checkpoints]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved checkpoint information to {output_file}")

    # Summary
    total_checkpoints = sum(len(checkpoints) for checkpoints in all_checkpoints.values())
    print(f"\nTotal checkpoints available: {total_checkpoints}")

    if all_checkpoints["pretraining"] and all_checkpoints["annealing"]:
        pre_max = max(step for step, _ in all_checkpoints["pretraining"])
        ann_min = min(step for step, _ in all_checkpoints["annealing"])
        ann_max = max(step for step, _ in all_checkpoints["annealing"])

        print(f"Pretraining steps: 0 - {pre_max}")
        print(f"Annealing steps: {ann_min} - {ann_max}")

if __name__ == "__main__":
    main()