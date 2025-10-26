#!/usr/bin/env python3
"""
Create the correct checkpoint sequence based on training progression:
1. All pretraining checkpoints first (steps 1192 → 119209)
2. All annealing checkpoints second (steps 1192 → 11921, but they come AFTER pretraining)
"""

import json
from typing import List, Tuple

def create_correct_checkpoint_sequence() -> List[dict]:
    """
    Create correctly ordered checkpoint sequence.
    Returns list of checkpoint dicts with correct temporal ordering.
    """

    # Load the checkpoint data
    with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/available_checkpoints.json', 'r') as f:
        checkpoints_data = json.load(f)

    corrected_checkpoints = []

    # Phase 1: Add ALL pretraining checkpoints first (these come first in time)
    print("Phase 1: Adding pretraining checkpoints...")
    for checkpoint in sorted(checkpoints_data["pretraining"], key=lambda x: x["step"]):
        corrected_checkpoints.append({
            "step": checkpoint["step"],
            "model_name": "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
            "revision": checkpoint["revision"],
            "stage": "pretraining",
            "temporal_order": len(corrected_checkpoints)
        })

    print(f"Added {len(corrected_checkpoints)} pretraining checkpoints")
    print(f"Pretraining range: {corrected_checkpoints[0]['step']} → {corrected_checkpoints[-1]['step']}")

    # Phase 2: Add ALL annealing checkpoints second (these come after ALL pretraining)
    print("\nPhase 2: Adding annealing checkpoints...")
    annealing_start_idx = len(corrected_checkpoints)

    for checkpoint in sorted(checkpoints_data["annealing"], key=lambda x: x["step"]):
        corrected_checkpoints.append({
            "step": checkpoint["step"],
            "model_name": "EleutherAI/deep-ignorance-unfiltered",
            "revision": checkpoint["revision"],
            "stage": "annealing",
            "temporal_order": len(corrected_checkpoints)
        })

    annealing_checkpoints = corrected_checkpoints[annealing_start_idx:]
    print(f"Added {len(annealing_checkpoints)} annealing checkpoints")
    print(f"Annealing range: {annealing_checkpoints[0]['step']} → {annealing_checkpoints[-1]['step']}")

    print(f"\nTotal checkpoints: {len(corrected_checkpoints)}")

    return corrected_checkpoints

def validate_sequence(checkpoints: List[dict]):
    """Validate the checkpoint sequence."""
    print("\n=== VALIDATION ===")

    # Check pretraining comes first
    pretraining_checkpoints = [ckpt for ckpt in checkpoints if ckpt['stage'] == 'pretraining']
    annealing_checkpoints = [ckpt for ckpt in checkpoints if ckpt['stage'] == 'annealing']

    last_pretraining_idx = max(ckpt['temporal_order'] for ckpt in pretraining_checkpoints)
    first_annealing_idx = min(ckpt['temporal_order'] for ckpt in annealing_checkpoints)

    print(f"Last pretraining checkpoint: temporal_order {last_pretraining_idx}")
    print(f"First annealing checkpoint: temporal_order {first_annealing_idx}")

    if first_annealing_idx > last_pretraining_idx:
        print("✅ Sequence is correct: All pretraining comes before all annealing")
    else:
        print("❌ ERROR: Sequence is wrong!")

    # Show first few and last few
    print("\nFirst 5 checkpoints:")
    for i, ckpt in enumerate(checkpoints[:5]):
        print(f"  {i}: {ckpt['stage']} step {ckpt['step']} ({ckpt['model_name'].split('/')[-1]})")

    print("\nLast 5 checkpoints:")
    for i, ckpt in enumerate(checkpoints[-5:], len(checkpoints)-5):
        print(f"  {i}: {ckpt['stage']} step {ckpt['step']} ({ckpt['model_name'].split('/')[-1]})")

def save_corrected_sequence(checkpoints: List[dict]):
    """Save the corrected checkpoint sequence."""
    output_file = '/mnt/ssd-1/lucia/deep-ignorance/analysis/corrected_checkpoints.json'
    with open(output_file, 'w') as f:
        json.dump(checkpoints, f, indent=2)
    print(f"\n✅ Saved corrected sequence to {output_file}")

def main():
    print("Creating correct checkpoint sequence...")
    print("Training progression: Pretraining (1192→119209) → Annealing (1192→11921)")

    # Create corrected sequence
    corrected_checkpoints = create_correct_checkpoint_sequence()

    # Validate
    validate_sequence(corrected_checkpoints)

    # Save
    save_corrected_sequence(corrected_checkpoints)

if __name__ == "__main__":
    main()