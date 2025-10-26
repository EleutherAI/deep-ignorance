#!/usr/bin/env python3
"""
Process and sort all available checkpoints for binary search analysis.
"""

import json
from typing import List, Tuple, Dict

def load_checkpoints() -> Dict[str, List[Dict]]:
    """Load checkpoint data from file."""
    with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/available_checkpoints.json', 'r') as f:
        return json.load(f)

def create_sorted_checkpoint_list(checkpoints_data: Dict[str, List[Dict]]) -> List[Tuple[int, str, str]]:
    """
    Create a single sorted list of all checkpoints.
    Returns list of (step, model_name, revision) tuples.
    """
    all_checkpoints = []

    # Add pretraining checkpoints
    for checkpoint in checkpoints_data["pretraining"]:
        step = checkpoint["step"]
        revision = checkpoint["revision"]
        model_name = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
        all_checkpoints.append((step, model_name, revision))

    # Add annealing checkpoints
    for checkpoint in checkpoints_data["annealing"]:
        step = checkpoint["step"]
        revision = checkpoint["revision"]
        model_name = "EleutherAI/deep-ignorance-unfiltered"
        all_checkpoints.append((step, model_name, revision))

    # Sort by step number
    all_checkpoints.sort(key=lambda x: x[0])

    return all_checkpoints

def analyze_checkpoint_progression(checkpoints: List[Tuple[int, str, str]]):
    """Analyze the checkpoint progression."""
    pretraining_checkpoints = [ckpt for ckpt in checkpoints if "pretraining-stage" in ckpt[1]]
    annealing_checkpoints = [ckpt for ckpt in checkpoints if "pretraining-stage" not in ckpt[1]]

    print("Checkpoint Analysis:")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Pretraining checkpoints: {len(pretraining_checkpoints)}")
    print(f"Annealing checkpoints: {len(annealing_checkpoints)}")

    if pretraining_checkpoints:
        pre_min = min(step for step, _, _ in pretraining_checkpoints)
        pre_max = max(step for step, _, _ in pretraining_checkpoints)
        print(f"Pretraining range: {pre_min} - {pre_max}")

    if annealing_checkpoints:
        ann_min = min(step for step, _, _ in annealing_checkpoints)
        ann_max = max(step for step, _, _ in annealing_checkpoints)
        print(f"Annealing range: {ann_min} - {ann_max}")

    print("\nFirst 10 checkpoints:")
    for i, (step, model, revision) in enumerate(checkpoints[:10]):
        model_short = "pretraining" if "pretraining-stage" in model else "annealing"
        print(f"  {i+1:2d}. Step {step:6d} ({model_short}): {revision}")

    print("\nLast 10 checkpoints:")
    for i, (step, model, revision) in enumerate(checkpoints[-10:], len(checkpoints)-9):
        model_short = "pretraining" if "pretraining-stage" in model else "annealing"
        print(f"  {i:2d}. Step {step:6d} ({model_short}): {revision}")

def save_processed_checkpoints(checkpoints: List[Tuple[int, str, str]]):
    """Save the processed checkpoint list."""
    # Convert to serializable format
    checkpoint_list = []
    for step, model_name, revision in checkpoints:
        checkpoint_list.append({
            "step": step,
            "model_name": model_name,
            "revision": revision,
            "stage": "pretraining" if "pretraining-stage" in model_name else "annealing"
        })

    output_file = '/mnt/ssd-1/lucia/deep-ignorance/analysis/sorted_checkpoints.json'
    with open(output_file, 'w') as f:
        json.dump(checkpoint_list, f, indent=2)

    print(f"\nSaved sorted checkpoints to {output_file}")
    return output_file

def main():
    # Load checkpoint data
    checkpoints_data = load_checkpoints()

    # Create sorted list
    sorted_checkpoints = create_sorted_checkpoint_list(checkpoints_data)

    # Analyze
    analyze_checkpoint_progression(sorted_checkpoints)

    # Save processed data
    save_processed_checkpoints(sorted_checkpoints)

if __name__ == "__main__":
    main()