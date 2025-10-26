#!/usr/bin/env python3
"""
Simple function to get the number of questions correct at each checkpoint.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def get_checkpoint_progression() -> List[Dict]:
    """
    Returns the number of questions the model gets right at each checkpoint.

    Returns:
        List of dictionaries, each containing:
        - checkpoint: checkpoint index (0-111)
        - step: training step number
        - stage: 'pretraining' or 'annealing'
        - temporal_order: true chronological order
        - correct_count: cumulative number of questions correct at this checkpoint
        - newly_correct: number of questions that became correct at this checkpoint
    """

    # Load data
    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results_corrected/progress.json")
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/corrected_checkpoints.json")
    with open(checkpoint_file, 'r') as f:
        checkpoints = json.load(f)

    # Create mapping for temporal orders
    step_to_temporal = {}
    for ckpt in checkpoints:
        key = f"{ckpt['stage']}_{ckpt['step']}"
        step_to_temporal[key] = ckpt['temporal_order']

    # Process questions
    always_correct = 0
    emerged_questions = []

    for question_key, result in emergence_data.items():
        if not result.get('search_complete', False):
            continue

        if result['always_correct']:
            always_correct += 1
        elif not result['never_correct']:
            # Question emerged during training
            emergence_step = result['emergence_step']
            emergence_stage = result.get('emergence_stage', 'unknown')
            emergence_key = f"{emergence_stage}_{emergence_step}"
            temporal_order = step_to_temporal.get(emergence_key)

            if temporal_order is not None:
                emerged_questions.append({
                    'temporal_order': temporal_order,
                    'step': emergence_step,
                    'stage': emergence_stage
                })

    # Calculate progression
    progression = []

    for i, ckpt in enumerate(checkpoints):
        current_temporal = ckpt['temporal_order']

        # Count questions correct at this checkpoint
        correct_count = always_correct
        newly_correct = 0

        for eq in emerged_questions:
            if eq['temporal_order'] <= current_temporal:
                correct_count += 1
            if eq['temporal_order'] == current_temporal:
                newly_correct += 1

        progression.append({
            'checkpoint': i,
            'step': ckpt['step'],
            'stage': ckpt['stage'],
            'temporal_order': current_temporal,
            'correct_count': correct_count,
            'newly_correct': newly_correct
        })

    return progression

def print_progression_summary():
    """Print a nice summary of the progression."""

    progression = get_checkpoint_progression()

    print("🎯 CHECKPOINT PROGRESSION: Questions Correct at Each Checkpoint")
    print("=" * 70)

    print(f"📊 Overview:")
    print(f"   • Total checkpoints: {len(progression)}")
    print(f"   • Starting correct: {progression[0]['correct_count']}")
    print(f"   • Final correct: {progression[-1]['correct_count']}")

    print(f"\n📈 Key Milestones:")

    # Show every 20th checkpoint plus first and last
    milestones = [0] + list(range(19, len(progression), 20)) + [len(progression)-1]

    for i in milestones:
        if i < len(progression):
            p = progression[i]
            stage_emoji = "🔷" if p['stage'] == 'pretraining' else "🔶"
            print(f"   {stage_emoji} Checkpoint {p['checkpoint']:3d}: {p['correct_count']:3d} correct "
                  f"({p['stage']:12s} step {p['step']:6d})")

    print(f"\n🚀 Growth Rate:")
    total_growth = progression[-1]['correct_count'] - progression[0]['correct_count']
    pretraining_end = next(i for i, p in enumerate(progression) if p['stage'] == 'annealing')
    pretraining_growth = progression[pretraining_end-1]['correct_count'] - progression[0]['correct_count']
    annealing_growth = progression[-1]['correct_count'] - progression[pretraining_end-1]['correct_count']

    print(f"   • Total questions gained: {total_growth}")
    print(f"   • During pretraining: {pretraining_growth}")
    print(f"   • During annealing: {annealing_growth}")

    return progression

def plot_checkpoint_progression(progression: List[Dict], save_pdf: bool = True):
    """
    Create a plot of the checkpoint progression and save to PDF and PNG.

    Args:
        progression: List of checkpoint data from get_checkpoint_progression()
        save_pdf: Whether to save PDF version (default True)
    """
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/final_deliverables")

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(progression)

    plt.figure(figsize=(15, 8))

    # Split by stage for different colors
    pretraining = df[df['stage'] == 'pretraining']
    annealing = df[df['stage'] == 'annealing']

    # Plot using temporal order for x-axis
    plt.plot(pretraining['temporal_order'], pretraining['correct_count'],
             'b-', linewidth=3, label='Pretraining', marker='o', markersize=4, alpha=0.8)

    if len(annealing) > 0:
        plt.plot(annealing['temporal_order'], annealing['correct_count'],
                 'r-', linewidth=3, label='Annealing', marker='s', markersize=4, alpha=0.8)

    plt.xlabel('Checkpoint (Temporal Order)', fontsize=12)
    plt.ylabel('Cumulative Correct Answers', fontsize=12)
    plt.title('Dangerous Capability Emergence Throughout Deep Ignorance Training\n' +
              'Cumulative Questions Answered Correctly by Checkpoint', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Add phase boundary
    if len(annealing) > 0:
        boundary = annealing['temporal_order'].iloc[0]
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(boundary - 5, plt.ylim()[1] * 0.9, 'Pretraining', ha='right', fontsize=10, alpha=0.7)
        plt.text(boundary + 1, plt.ylim()[1] * 0.9, 'Annealing', ha='left', fontsize=10, alpha=0.7)

    # Add annotations for key points
    start_count = progression[0]['correct_count']
    end_count = progression[-1]['correct_count']

    plt.annotate(f'Start: {start_count} questions',
                xy=(0, start_count), xytext=(10, start_count + 30),
                arrowprops=dict(arrowstyle='->', alpha=0.6), fontsize=10)

    plt.annotate(f'Final: {end_count} questions',
                xy=(len(progression)-1, end_count), xytext=(len(progression)-20, end_count - 30),
                arrowprops=dict(arrowstyle='->', alpha=0.6), fontsize=10)

    plt.tight_layout()

    # Save both PDF and PNG
    if save_pdf:
        plt.savefig(output_dir / 'checkpoint_progression.pdf', dpi=300, bbox_inches='tight')
        print(f"📄 PDF plot saved to: {output_dir}/checkpoint_progression.pdf")

    plt.savefig(output_dir / 'checkpoint_progression.png', dpi=300, bbox_inches='tight')
    print(f"🖼️ PNG plot saved to: {output_dir}/checkpoint_progression.png")

    plt.close()

if __name__ == "__main__":
    progression = print_progression_summary()

    # Create and save plot
    plot_checkpoint_progression(progression, save_pdf=True)

    # Also save to CSV for easy access
    df = pd.DataFrame(progression)
    output_file = "/mnt/ssd-1/lucia/deep-ignorance/analysis/final_deliverables/checkpoint_progression_simple.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")