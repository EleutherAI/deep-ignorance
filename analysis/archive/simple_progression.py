#!/usr/bin/env python3
"""
Simple function to show cumulative correct answers by training progression.
"""

import json
import pandas as pd
from pathlib import Path

def get_simple_progression():
    """Get a simple progression showing questions correct at each checkpoint."""

    # Load emergence data
    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results_corrected/progress.json")
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

    # Load checkpoints in correct temporal order
    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/corrected_checkpoints.json")
    with open(checkpoint_file, 'r') as f:
        checkpoints = json.load(f)

    print("=== CHECKPOINT PROGRESSION ANALYSIS ===")
    print(f"Total questions: {len(emergence_data)}")
    print(f"Total checkpoints: {len(checkpoints)}")

    # Count categories
    always_correct = 0
    never_correct = 0
    emerged_questions = []

    for question_key, result in emergence_data.items():
        if not result.get('search_complete', False):
            continue

        if result['always_correct']:
            always_correct += 1
        elif result['never_correct']:
            never_correct += 1
        else:
            # Question emerged at specific step
            if result['emergence_step']:
                emerged_questions.append({
                    'question': question_key,
                    'step': result['emergence_step'],
                    'stage': result.get('emergence_stage', 'unknown')
                })

    print(f"\nQuestion categories:")
    print(f"- Always correct: {always_correct}")
    print(f"- Never correct: {never_correct}")
    print(f"- Emerged during training: {len(emerged_questions)}")

    # Calculate progression
    progression = []

    for i, ckpt in enumerate(checkpoints):
        step = ckpt['step']
        stage = ckpt['stage']

        # Count correct at this checkpoint
        correct_count = always_correct  # Always correct from start

        # Add questions that emerged at or before this step
        for eq in emerged_questions:
            if eq['step'] <= step:
                correct_count += 1

        progression.append({
            'checkpoint': i,
            'step': step,
            'stage': stage,
            'temporal_order': ckpt['temporal_order'],
            'correct_count': correct_count
        })

    # Show key milestones
    print(f"\n=== KEY MILESTONES ===")

    milestones = [0, 25, 50, 75, 100, len(checkpoints)-1]
    for i in milestones:
        if i < len(progression):
            p = progression[i]
            print(f"Checkpoint {p['checkpoint']:3d} | Step {p['step']:6d} | {p['stage']:12s} | {p['correct_count']:3d} questions correct")

    # Show final numbers
    final_count = progression[-1]['correct_count']
    total_possible = always_correct + len(emerged_questions)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Final correct answers: {final_count}")
    print(f"Total possible: {total_possible}")
    print(f"Never correct: {never_correct}")
    print(f"Total questions: {len(emergence_data)}")

    return progression

if __name__ == "__main__":
    progression = get_simple_progression()