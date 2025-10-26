#!/usr/bin/env python3
"""
Function to show the number of questions the model gets right at each checkpoint.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_emergence_data() -> Dict:
    """Load the emergence analysis results."""
    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results_corrected/progress.json")

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def load_checkpoint_data() -> List[Dict]:
    """Load the checkpoint sequence data."""
    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/corrected_checkpoints.json")

    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return []

def calculate_cumulative_correct_answers(emergence_data: Dict, checkpoints: List[Dict]) -> pd.DataFrame:
    """
    Calculate the cumulative number of correct answers at each checkpoint.

    Returns a DataFrame with columns:
    - checkpoint_idx: Index of the checkpoint
    - step: Training step number
    - stage: Training stage (pretraining/annealing)
    - temporal_order: Temporal order in training
    - cumulative_correct: Number of questions correct at this checkpoint
    - newly_correct: Number of questions that became correct at this checkpoint
    """

    # Create a mapping of steps to checkpoint indices
    step_to_checkpoint = {}
    for i, ckpt in enumerate(checkpoints):
        step_to_checkpoint[ckpt['step']] = i

    # Process emergence data
    emergence_steps = []
    always_correct_count = 0
    never_correct_count = 0

    for question_key, result in emergence_data.items():
        if not result.get('search_complete', False):
            continue

        if result['always_correct']:
            always_correct_count += 1
        elif result['never_correct']:
            never_correct_count += 1
        else:
            # Question emerged at a specific step
            emergence_step = result['emergence_step']
            if emergence_step:
                emergence_steps.append(emergence_step)

    print(f"Total questions analyzed: {len(emergence_data)}")
    print(f"Always correct: {always_correct_count}")
    print(f"Never correct: {never_correct_count}")
    print(f"Emerged during training: {len(emergence_steps)}")

    # Create results dataframe
    results = []

    for i, ckpt in enumerate(checkpoints):
        step = ckpt['step']
        stage = ckpt['stage']
        temporal_order = ckpt['temporal_order']

        # Count questions correct at this checkpoint
        # 1. Always correct questions are correct from the start
        cumulative_correct = always_correct_count

        # 2. Add questions that emerged at or before this step
        newly_correct_at_this_step = 0
        for emergence_step in emergence_steps:
            if emergence_step <= step:
                cumulative_correct += 1
            if emergence_step == step:
                newly_correct_at_this_step += 1

        results.append({
            'checkpoint_idx': i,
            'step': step,
            'stage': stage,
            'temporal_order': temporal_order,
            'cumulative_correct': cumulative_correct,
            'newly_correct': newly_correct_at_this_step
        })

    return pd.DataFrame(results)

def plot_emergence_progression(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of the emergence progression."""

    # Create the cumulative progression plot
    plt.figure(figsize=(15, 8))

    # Split by stage for different colors
    pretraining_data = df[df['stage'] == 'pretraining']
    annealing_data = df[df['stage'] == 'annealing']

    # Plot cumulative correct answers
    plt.plot(pretraining_data['step'], pretraining_data['cumulative_correct'],
             'b-', linewidth=2, label='Pretraining', alpha=0.8)

    if len(annealing_data) > 0:
        plt.plot(annealing_data['step'], annealing_data['cumulative_correct'],
                 'r-', linewidth=2, label='Annealing', alpha=0.8)

    plt.xlabel('Training Step')
    plt.ylabel('Cumulative Number of Correct Answers')
    plt.title('Question Emergence Throughout Training\n(Cumulative Correct Answers by Checkpoint)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations for key milestones
    final_correct = df['cumulative_correct'].iloc[-1]
    plt.axhline(y=final_correct, color='gray', linestyle='--', alpha=0.5)
    plt.text(df['step'].iloc[-1] * 0.7, final_correct + 5,
             f'Final: {final_correct} questions', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_emergence_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a "new emergences" plot showing the rate of emergence
    plt.figure(figsize=(15, 6))

    # Only plot points where newly_correct > 0
    emergence_points = df[df['newly_correct'] > 0]

    plt.scatter(emergence_points['step'], emergence_points['newly_correct'],
                c=['blue' if stage == 'pretraining' else 'red' for stage in emergence_points['stage']],
                alpha=0.7, s=50)

    plt.xlabel('Training Step')
    plt.ylabel('Number of Questions Newly Correct')
    plt.title('Rate of Question Emergence Throughout Training')
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.scatter([], [], c='blue', label='Pretraining')
    plt.scatter([], [], c='red', label='Annealing')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'emergence_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")

def save_progression_data(df: pd.DataFrame, output_dir: Path):
    """Save the progression data to files."""

    # Save detailed progression
    progression_file = output_dir / 'checkpoint_progression.csv'
    df.to_csv(progression_file, index=False)
    print(f"Detailed progression saved to {progression_file}")

    # Create a simplified summary
    summary = {
        'total_checkpoints': len(df),
        'final_correct_answers': int(df['cumulative_correct'].iloc[-1]),
        'pretraining_checkpoints': len(df[df['stage'] == 'pretraining']),
        'annealing_checkpoints': len(df[df['stage'] == 'annealing']),
        'steps_with_emergence': len(df[df['newly_correct'] > 0]),
        'progression_by_checkpoint': [
            {
                'checkpoint': int(row['checkpoint_idx']),
                'step': int(row['step']),
                'stage': row['stage'],
                'cumulative_correct': int(row['cumulative_correct']),
                'newly_correct': int(row['newly_correct'])
            }
            for _, row in df.iterrows()
        ]
    }

    summary_file = output_dir / 'checkpoint_progression_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

def get_cumulative_correct_answers() -> pd.DataFrame:
    """
    Main function to calculate and return the cumulative correct answers at each checkpoint.

    Returns:
        DataFrame with checkpoint progression data
    """
    print("Loading emergence data and checkpoints...")

    # Load data
    emergence_data = load_emergence_data()
    checkpoints = load_checkpoint_data()

    if not emergence_data:
        raise ValueError("No emergence data found. Make sure the analysis has completed.")

    if not checkpoints:
        raise ValueError("No checkpoint data found.")

    print(f"Loaded {len(emergence_data)} questions and {len(checkpoints)} checkpoints")

    # Calculate progression
    df = calculate_cumulative_correct_answers(emergence_data, checkpoints)

    return df

def main():
    """Main function to generate the complete checkpoint progression analysis."""

    # Get the progression data
    df = get_cumulative_correct_answers()

    # Create output directory
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/final_deliverables")
    output_dir.mkdir(exist_ok=True)

    # Create visualizations
    plot_emergence_progression(df, output_dir)

    # Save data
    save_progression_data(df, output_dir)

    # Print summary statistics
    print("\n" + "="*60)
    print("CHECKPOINT PROGRESSION SUMMARY")
    print("="*60)
    print(f"Total checkpoints: {len(df)}")
    print(f"Final correct answers: {df['cumulative_correct'].iloc[-1]}")
    print(f"\nKey milestones:")

    # Show some key milestones
    milestones = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    for i in milestones:
        row = df.iloc[i]
        print(f"  Checkpoint {row['checkpoint_idx']:3d} (step {row['step']:6d}, {row['stage']}): {row['cumulative_correct']:3d} correct")

    print(f"\nFiles saved to: {output_dir}")
    print("- checkpoint_progression.csv: Detailed data")
    print("- checkpoint_progression_summary.json: Summary statistics")
    print("- cumulative_emergence_progression.png: Main visualization")
    print("- emergence_rate.png: Rate of emergence")

if __name__ == "__main__":
    main()