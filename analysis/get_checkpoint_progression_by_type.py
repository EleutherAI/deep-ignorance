#!/usr/bin/env python3
"""
Function to show the number of questions the model gets right at each checkpoint,
split by question type (cloze vs MCQA).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def get_checkpoint_progression_by_type(tokens_per_step: int) -> Dict[str, List[Dict]]:
    """
    Returns the number of questions the model gets right at each checkpoint,
    split by question type (cloze vs MCQA).

    Returns:
        Dictionary with keys 'cloze' and 'mcqa', each containing a list of:
        - checkpoint: checkpoint index (0-111)
        - step: training step number
        - all_stages_step: training step number accumulated over all stages
        - stage: 'pretraining' or 'annealing'
        - temporal_order: true chronological order
        - correct_count: cumulative number of questions correct at this checkpoint
        - newly_correct: number of questions that became correct at this checkpoint
    """

    # Load data
    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/emergence_results_corrected/final_results.json")
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/corrected_checkpoints.json")
    with open(checkpoint_file, 'r') as f:
        checkpoints = json.load(f)

    # Create mapping for temporal orders
    step_to_temporal = {}
    for ckpt in checkpoints:
        key = f"{ckpt['stage']}_{ckpt['step']}"
        step_to_temporal[key] = ckpt['temporal_order']

    # Separate questions by type
    cloze_questions = {'always_correct': 0, 'emerged': []}
    mcqa_questions = {'always_correct': 0, 'emerged': []}

    for question_key, result in emergence_data.items():
        if not result.get('search_complete', False):
            continue

        # Determine question type from task name
        task = result.get('task', '')
        is_cloze = 'cloze' in task.lower()

        if result['always_correct']:
            if is_cloze:
                cloze_questions['always_correct'] += 1
            else:
                mcqa_questions['always_correct'] += 1
        elif not result['never_correct']:
            # Question emerged during training
            emergence_step = result['emergence_step']
            emergence_stage = result.get('emergence_stage', 'unknown')
            emergence_key = f"{emergence_stage}_{emergence_step}"
            temporal_order = step_to_temporal.get(emergence_key)

            if temporal_order is not None:
                question_info = {
                    'temporal_order': temporal_order,
                    'step': emergence_step,
                    'stage': emergence_stage
                }

                if is_cloze:
                    cloze_questions['emerged'].append(question_info)
                else:
                    mcqa_questions['emerged'].append(question_info)

    # Calculate progression for each type
    def calculate_progression(question_data, tokens_per_step):
        progression = []
        always_correct = question_data['always_correct']
        emerged_questions = question_data['emerged']
        
        pretraining_final_step = max(
            ckpt['step'] 
            for ckpt in checkpoints 
            if ckpt['stage'] == 'pretraining'
        )

        for i, ckpt in enumerate(checkpoints):
            current_temporal = ckpt['temporal_order']
            
            all_stages_step = (
                ckpt['step'] 
                if ckpt['stage'] == 'pretraining' 
                else ckpt['step'] + pretraining_final_step
            )

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
                'all_stages_step': all_stages_step,
                'all_stages_tokens': all_stages_step * tokens_per_step,
                'stage': ckpt['stage'],
                'temporal_order': current_temporal,
                'correct_count': correct_count,
                'newly_correct': newly_correct
            })

        return progression

    cloze_progression = calculate_progression(cloze_questions, tokens_per_step)
    mcqa_progression = calculate_progression(mcqa_questions, tokens_per_step)

    return {
        'cloze': cloze_progression,
        'mcqa': mcqa_progression
    }

def print_progression_summary_by_type(tokens_per_step):
    """Print a nice summary of the progression by question type."""

    progressions = get_checkpoint_progression_by_type(tokens_per_step)
    cloze_prog = progressions['cloze']
    mcqa_prog = progressions['mcqa']

    print("🎯 CHECKPOINT PROGRESSION BY QUESTION TYPE")
    print("=" * 70)

    print(f"📊 Overview:")
    print(f"   • Total checkpoints: {len(cloze_prog)}")
    print(f"   • Cloze questions: {cloze_prog[0]['correct_count']} → {cloze_prog[-1]['correct_count']}")
    print(f"   • MCQA questions: {mcqa_prog[0]['correct_count']} → {mcqa_prog[-1]['correct_count']}")
    print(f"   • Combined total: {cloze_prog[0]['correct_count'] + mcqa_prog[0]['correct_count']} → {cloze_prog[-1]['correct_count'] + mcqa_prog[-1]['correct_count']}")

    print(f"\n📈 Key Milestones:")

    # Show every 20th checkpoint plus first and last
    milestones = [0] + list(range(19, len(cloze_prog), 20)) + [len(cloze_prog)-1]

    for i in milestones:
        if i < len(cloze_prog):
            c_p = cloze_prog[i]
            m_p = mcqa_prog[i]
            stage_emoji = "🔷" if c_p['stage'] == 'pretraining' else "🔶"
            print(f"   {stage_emoji} Checkpoint {c_p['checkpoint']:3d}: Cloze {c_p['correct_count']:3d}, MCQA {m_p['correct_count']:3d} "
                  f"({c_p['stage']:12s} step {c_p['step']:6d})")

    print(f"\n🚀 Growth Rate:")
    cloze_growth = cloze_prog[-1]['correct_count'] - cloze_prog[0]['correct_count']
    mcqa_growth = mcqa_prog[-1]['correct_count'] - mcqa_prog[0]['correct_count']

    # Find pretraining end
    pretraining_end = next(i for i, p in enumerate(cloze_prog) if p['stage'] == 'annealing')

    cloze_pretraining_growth = cloze_prog[pretraining_end-1]['correct_count'] - cloze_prog[0]['correct_count']
    cloze_annealing_growth = cloze_prog[-1]['correct_count'] - cloze_prog[pretraining_end-1]['correct_count']

    mcqa_pretraining_growth = mcqa_prog[pretraining_end-1]['correct_count'] - mcqa_prog[0]['correct_count']
    mcqa_annealing_growth = mcqa_prog[-1]['correct_count'] - mcqa_prog[pretraining_end-1]['correct_count']

    print(f"   • Cloze questions gained: {cloze_growth} (pretraining: {cloze_pretraining_growth}, annealing: {cloze_annealing_growth})")
    print(f"   • MCQA questions gained: {mcqa_growth} (pretraining: {mcqa_pretraining_growth}, annealing: {mcqa_annealing_growth})")

    return progressions

def plot_checkpoint_progression_by_type(progressions: Dict[str, List[Dict]], save_pdf: bool = True):
    """
    Create a plot of the checkpoint progression split by question type and save to PDF and PNG.

    Args:
        progressions: Dictionary with 'cloze' and 'mcqa' progression data
        save_pdf: Whether to save PDF version (default True)
    """
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/final_deliverables")

    # Convert to DataFrames for easier plotting
    cloze_df = pd.DataFrame(progressions['cloze'])
    mcqa_df = pd.DataFrame(progressions['mcqa'])

    plt.figure(figsize=(15, 8))

    # Split by stage for different line styles
    cloze_pretraining = cloze_df[cloze_df['stage'] == 'pretraining']
    cloze_annealing = cloze_df[cloze_df['stage'] == 'annealing']
    mcqa_pretraining = mcqa_df[mcqa_df['stage'] == 'pretraining']
    mcqa_annealing = mcqa_df[mcqa_df['stage'] == 'annealing']

    # Plot cloze questions
    plt.plot(cloze_pretraining['temporal_order'], cloze_pretraining['correct_count'],
             'b-', linewidth=3, label='Cloze (Pretraining)', marker='o', markersize=3, alpha=0.8)

    if len(cloze_annealing) > 0:
        plt.plot(cloze_annealing['temporal_order'], cloze_annealing['correct_count'],
                 'b--', linewidth=3, label='Cloze (Annealing)', marker='o', markersize=3, alpha=0.8)

    # Plot MCQA questions
    plt.plot(mcqa_pretraining['temporal_order'], mcqa_pretraining['correct_count'],
             'r-', linewidth=3, label='MCQA (Pretraining)', marker='s', markersize=3, alpha=0.8)

    if len(mcqa_annealing) > 0:
        plt.plot(mcqa_annealing['temporal_order'], mcqa_annealing['correct_count'],
                 'r--', linewidth=3, label='MCQA (Annealing)', marker='s', markersize=3, alpha=0.8)

    plt.xlabel('Checkpoint (Temporal Order)', fontsize=12)
    plt.ylabel('Cumulative Correct Answers', fontsize=12)
    plt.title('Dangerous Capability Emergence by Question Type\\n' +
              'Cumulative Questions Answered Correctly by Checkpoint', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')

    # Add phase boundary
    if len(cloze_annealing) > 0:
        boundary = cloze_annealing['temporal_order'].iloc[0]
        plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        plt.text(boundary - 8, plt.ylim()[1] * 0.95, 'Pretraining', ha='right', fontsize=10, alpha=0.7)
        plt.text(boundary + 1, plt.ylim()[1] * 0.95, 'Annealing', ha='left', fontsize=10, alpha=0.7)

    # Add annotations for final counts
    cloze_final = progressions['cloze'][-1]['correct_count']
    mcqa_final = progressions['mcqa'][-1]['correct_count']
    total_final = cloze_final + mcqa_final
    breakpoint()

    plt.annotate(f'Final Cloze: {cloze_final}',
                xy=(len(progressions['cloze'])-1, cloze_final),
                xytext=(len(progressions['cloze'])-15, cloze_final + 20),
                arrowprops=dict(arrowstyle='->', alpha=0.6, color='blue'),
                fontsize=10, color='blue')

    plt.annotate(f'Final MCQA: {mcqa_final}',
                xy=(len(progressions['mcqa'])-1, mcqa_final),
                xytext=(len(progressions['mcqa'])-15, mcqa_final - 30),
                arrowprops=dict(arrowstyle='->', alpha=0.6, color='red'),
                fontsize=10, color='red')

    plt.tight_layout()

    # Save both PDF and PNG
    if save_pdf:
        plt.savefig(output_dir / 'checkpoint_progression_by_type.pdf', dpi=300, bbox_inches='tight')
        print(f"📄 PDF plot saved to: {output_dir}/checkpoint_progression_by_type.pdf")

    plt.savefig(output_dir / 'checkpoint_progression_by_type.png', dpi=300, bbox_inches='tight')
    print(f"🖼️ PNG plot saved to: {output_dir}/checkpoint_progression_by_type.png")

    plt.close()

if __name__ == "__main__":
    tokens_per_step = 4194304

    progressions = print_progression_summary_by_type(tokens_per_step)

    # Create and save plot
    plot_checkpoint_progression_by_type(progressions, save_pdf=True)

    # Also save to CSV for easy access
    cloze_df = pd.DataFrame(progressions['cloze'])
    mcqa_df = pd.DataFrame(progressions['mcqa'])

    cloze_output = "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/final_deliverables/checkpoint_progression_cloze.csv"
    mcqa_output = "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/final_deliverables/checkpoint_progression_mcqa.csv"

    cloze_df.to_csv(cloze_output, index=False)
    mcqa_df.to_csv(mcqa_output, index=False)

    print(f"\n💾 Cloze data saved to: {cloze_output}")
    print(f"💾 MCQA data saved to: {mcqa_output}")