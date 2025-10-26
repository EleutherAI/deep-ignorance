#!/usr/bin/env python3
"""
Fix checkpoint ordering to reflect correct training progression:
1. Pretraining stage first (1192 → 119209)
2. Annealing stage second (continuing from pretraining)
"""

import json
from typing import List, Tuple, Dict

def create_correct_checkpoint_sequence() -> List[Tuple[int, str, str, str]]:
    """
    Create correctly ordered checkpoint sequence.
    Returns list of (global_step, model_name, revision, stage) tuples.
    """

    # Load the checkpoint data
    with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/available_checkpoints.json', 'r') as f:
        checkpoints_data = json.load(f)

    all_checkpoints = []

    # 1. Add all pretraining checkpoints first
    for checkpoint in checkpoints_data["pretraining"]:
        step = checkpoint["step"]
        revision = checkpoint["revision"]
        model_name = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
        all_checkpoints.append((step, model_name, revision, "pretraining"))

    # 2. Add annealing checkpoints, but they continue from pretraining
    # The annealing model step numbers likely need to be offset
    pretraining_final_step = max(ckpt["step"] for ckpt in checkpoints_data["pretraining"])

    print(f"Pretraining final step: {pretraining_final_step}")
    print(f"Annealing checkpoints: {[ckpt['step'] for ckpt in checkpoints_data['annealing']]}")

    # The annealing checkpoints appear to have their own step numbering
    # We need to understand how they relate to pretraining
    for checkpoint in checkpoints_data["annealing"]:
        step = checkpoint["step"]
        revision = checkpoint["revision"]
        model_name = "EleutherAI/deep-ignorance-unfiltered"

        # These are likely continuation steps after pretraining
        # The step numbers in annealing model might be global steps that continue from pretraining
        # OR they might be separate annealing-only steps

        # For now, let's assume they continue from pretraining
        # We'll need to clarify this with the user
        all_checkpoints.append((step + pretraining_final_step, model_name, revision, "annealing"))

    # Sort by step number to get correct temporal order
    all_checkpoints.sort(key=lambda x: x[0])

    return all_checkpoints

def analyze_checkpoint_progression():
    """Analyze the checkpoint progression to understand the timeline."""

    with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/available_checkpoints.json', 'r') as f:
        data = json.load(f)

    print("=== CHECKPOINT ANALYSIS ===")
    print(f"Pretraining checkpoints: {len(data['pretraining'])}")
    print(f"Range: {min(ckpt['step'] for ckpt in data['pretraining'])} → {max(ckpt['step'] for ckpt in data['pretraining'])}")

    print(f"\nAnnealing checkpoints: {len(data['annealing'])}")
    print(f"Range: {min(ckpt['step'] for ckpt in data['annealing'])} → {max(ckpt['step'] for ckpt in data['annealing'])}")

    print("\n=== QUESTION FOR USER ===")
    print("The annealing checkpoint step numbers (1192-11921) overlap with pretraining (1192-119209).")
    print("Are the annealing steps:")
    print("A) Continuation steps that come AFTER pretraining step 119209?")
    print("B) Separate numbering that resets for the annealing phase?")
    print("C) Something else?")

    print("\nWe need to know this to create the correct temporal ordering for binary search.")

def main():
    print("Analyzing checkpoint progression...")
    analyze_checkpoint_progression()

    print("\n" + "="*50)
    print("CURRENT ISSUE:")
    print("Need clarification on how annealing step numbers relate to pretraining")
    print("before proceeding with the corrected binary search.")

if __name__ == "__main__":
    main()