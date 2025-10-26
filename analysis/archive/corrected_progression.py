#!/usr/bin/env python3
"""
Corrected progression function using proper temporal ordering.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def get_corrected_progression():
    """Get progression using proper temporal ordering."""

    # Load emergence data
    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results_corrected/progress.json")
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

    # Load checkpoints in correct temporal order
    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/corrected_checkpoints.json")
    with open(checkpoint_file, 'r') as f:
        checkpoints = json.load(f)

    print("=== CORRECTED CHECKPOINT PROGRESSION ANALYSIS ===")
    print(f"Total questions: {len(emergence_data)}")
    print(f"Total checkpoints: {len(checkpoints)}")

    # Create step-to-temporal-order mapping
    step_to_temporal = {}
    for ckpt in checkpoints:
        key = f"{ckpt['stage']}_{ckpt['step']}"  # Use stage + step as key
        step_to_temporal[key] = ckpt['temporal_order']

    # Count categories and get emergence info
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
            # Question emerged at specific checkpoint
            emergence_step = result['emergence_step']
            emergence_stage = result.get('emergence_stage', 'unknown')

            # Find the temporal order for this emergence
            emergence_key = f"{emergence_stage}_{emergence_step}"
            temporal_order = step_to_temporal.get(emergence_key, None)

            if temporal_order is not None:
                emerged_questions.append({
                    'question': question_key,
                    'step': emergence_step,
                    'stage': emergence_stage,
                    'temporal_order': temporal_order
                })

    print(f"\nQuestion categories:")
    print(f"- Always correct: {always_correct}")
    print(f"- Never correct: {never_correct}")
    print(f"- Emerged during training: {len(emerged_questions)}")

    # Calculate corrected progression using temporal order
    progression = []

    for i, ckpt in enumerate(checkpoints):
        current_temporal_order = ckpt['temporal_order']

        # Count correct at this checkpoint
        correct_count = always_correct  # Always correct from start

        # Add questions that emerged at or before this temporal position
        for eq in emerged_questions:
            if eq['temporal_order'] <= current_temporal_order:
                correct_count += 1

        progression.append({
            'checkpoint': i,
            'step': ckpt['step'],
            'stage': ckpt['stage'],
            'temporal_order': current_temporal_order,
            'correct_count': correct_count
        })

    # Show progression throughout training
    print(f"\n=== TRAINING PROGRESSION ===")

    # Show key milestones
    milestones = [0, 20, 40, 60, 80, 100, len(checkpoints)-1]
    for i in milestones:
        if i < len(progression):
            p = progression[i]
            print(f"Checkpoint {p['checkpoint']:3d} | {p['stage']:12s} Step {p['step']:6d} | Temporal {p['temporal_order']:3d} | {p['correct_count']:3d} questions correct")

    # Show the actual training progression (first 5 and last 5)
    print(f"\n=== FIRST 5 CHECKPOINTS ===")
    for i in range(min(5, len(progression))):
        p = progression[i]
        print(f"  {p['checkpoint']:2d}: {p['stage']} step {p['step']} → {p['correct_count']} correct")

    print(f"\n=== LAST 5 CHECKPOINTS ===")
    for i in range(max(0, len(progression)-5), len(progression)):
        p = progression[i]
        print(f"  {p['checkpoint']:2d}: {p['stage']} step {p['step']} → {p['correct_count']} correct")

    # Final summary
    final_count = progression[-1]['correct_count']
    max_count = max(p['correct_count'] for p in progression)

    print(f"\n=== SUMMARY ===")
    print(f"Peak correct answers: {max_count}")
    print(f"Final correct answers: {final_count}")
    print(f"Total questions analyzed: {len(emergence_data)}")

    return progression

def plot_corrected_progression(progression):
    """Create a plot of the corrected progression."""

    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/final_deliverables")

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(progression)

    plt.figure(figsize=(15, 8))

    # Split by stage
    pretraining = df[df['stage'] == 'pretraining']
    annealing = df[df['stage'] == 'annealing']

    # Plot using temporal order for x-axis
    plt.plot(pretraining['temporal_order'], pretraining['correct_count'],
             'b-', linewidth=2, label='Pretraining', marker='o', markersize=3)

    if len(annealing) > 0:
        plt.plot(annealing['temporal_order'], annealing['correct_count'],
                 'r-', linewidth=2, label='Annealing', marker='s', markersize=3)

    plt.xlabel('Temporal Order (Checkpoint Index)')
    plt.ylabel('Cumulative Number of Correct Answers')
    plt.title('Cumulative Correct Answers Throughout Training\n(Using Correct Temporal Ordering)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add phase boundary
    if len(annealing) > 0:
        boundary = annealing['temporal_order'].iloc[0]
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7,
                   label='Pretraining → Annealing')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'corrected_cumulative_progression.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'corrected_cumulative_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_dir}/corrected_cumulative_progression.pdf and .png")

if __name__ == "__main__":
    progression = get_corrected_progression()
    plot_corrected_progression(progression)