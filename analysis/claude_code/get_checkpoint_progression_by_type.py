#!/usr/bin/env python3
"""
Function to show the number of questions the model gets right at each checkpoint,
split by question type (cloze vs MCQA).
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import List, Dict


def get_checkpoint_progression_by_type(tokens_per_step: int, progress_file: Path, checkpoint_file: Path) -> Dict[str, List[Dict]]:
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
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

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

        total_correct = always_correct + len(emerged_questions)
        
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
                'newly_correct': newly_correct,
                'fraction_correct': correct_count / total_correct
            })

        return progression

    cloze_progression = calculate_progression(cloze_questions, tokens_per_step)
    mcqa_progression = calculate_progression(mcqa_questions, tokens_per_step)

    return {
        'cloze': cloze_progression,
        'mcqa': mcqa_progression
    }

def print_progression_summary_by_type(tokens_per_step, progress_file: Path, checkpoint_file: Path):
    """Print a nice summary of the progression by question type."""

    progressions = get_checkpoint_progression_by_type(tokens_per_step, progress_file, checkpoint_file)
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

def plot_checkpoint_progression_by_type(progressions: Dict[str, List[Dict]], output_path: Path, plot_fraction: bool = False):
    """
    Create a plot of the checkpoint progression split by question type and save to PDF and PNG.

    Args:
        progressions: Dictionary with 'cloze' and 'mcqa' progression data
        save_pdf: Whether to save PDF version (default True)
    """
    pdf_path = output_dir / f'checkpoint_progression_by_type_{"fraction" if plot_fraction else "count"}.pdf'

    # Convert to DataFrames for easier plotting
    cloze_df = pd.DataFrame(progressions['cloze'])
    mcqa_df = pd.DataFrame(progressions['mcqa'])

    plt.figure(figsize=(15, 8))

    # Split by stage for different line styles
    cloze_pretraining = cloze_df[cloze_df['stage'] == 'pretraining']
    cloze_annealing = cloze_df[cloze_df['stage'] == 'annealing']
    mcqa_pretraining = mcqa_df[mcqa_df['stage'] == 'pretraining']
    mcqa_annealing = mcqa_df[mcqa_df['stage'] == 'annealing']

    y_column = 'correct_count' if not plot_fraction else 'fraction_correct'
    if plot_fraction:
        if plot_fraction:
            plt.plot(cloze_pretraining['all_stages_tokens'], cloze_pretraining[y_column] * 100,
                    'b-', linewidth=3, label='Cloze (Pretraining)', marker='o', markersize=3, alpha=0.8)
            
            if len(cloze_annealing) > 0:
                plt.plot(cloze_annealing['all_stages_tokens'], cloze_annealing[y_column] * 100,
                        'b--', linewidth=3, label='Cloze (Annealing)', marker='o', markersize=3, alpha=0.8)
            
            plt.plot(mcqa_pretraining['all_stages_tokens'], mcqa_pretraining[y_column] * 100,
                    'r-', linewidth=3, label='MCQA (Pretraining)', marker='s', markersize=3, alpha=0.8)
            
            if len(mcqa_annealing) > 0:
                plt.plot(mcqa_annealing['all_stages_tokens'], mcqa_annealing[y_column] * 100,
                        'r--', linewidth=3, label='MCQA (Annealing)', marker='s', markersize=3, alpha=0.8)
            
            plt.ylabel('Percent Correct (%)', fontsize=12)
    else:
        # Keep your existing plotting code for count
        plt.plot(cloze_pretraining['all_stages_tokens'], cloze_pretraining[y_column],
                'b-', linewidth=3, label='Cloze (Pretraining)', marker='o', markersize=3, alpha=0.8)
       

        if len(cloze_annealing) > 0:
            plt.plot(cloze_annealing['all_stages_tokens'], cloze_annealing[y_column],
                    'b--', linewidth=3, label='Cloze (Annealing)', marker='o', markersize=3, alpha=0.8)

        # Plot MCQA questions
        plt.plot(mcqa_pretraining['all_stages_tokens'], mcqa_pretraining[y_column],
                'r-', linewidth=3, label='MCQA (Pretraining)', marker='s', markersize=3, alpha=0.8)

        if len(mcqa_annealing) > 0:
            plt.plot(mcqa_annealing['all_stages_tokens'], mcqa_annealing[y_column],
                    'r--', linewidth=3, label='MCQA (Annealing)', marker='s', markersize=3, alpha=0.8)

        plt.ylabel('Cumulative Correct Answers', fontsize=12)

    plt.xlabel('Tokens Trained', fontsize=12)
    
    plt.title('Cumulative Learning of Questions Answered Correctly At Final Checkpoint', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')

    min_tok = cloze_df['all_stages_tokens'].min()
    max_tok = cloze_df['all_stages_tokens'].max()
    print(min_tok, max_tok)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1e9)}B'))

    # Set the x-axis limits to your actual data range
    all_tokens = pd.concat([cloze_df['all_stages_tokens'], mcqa_df['all_stages_tokens']])
    min_tok, max_tok = all_tokens.min(), all_tokens.max()
    ax.set_xlim(min_tok, max_tok)

    # Add phase boundary
    if len(cloze_annealing) > 0:
        boundary = cloze_annealing['all_stages_tokens'].iloc[0]
        plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        plt.text(boundary - 10, plt.ylim()[1] * 0.30, 'Pretraining ', ha='right', fontsize=10, alpha=0.7)
        plt.text(boundary + 10, plt.ylim()[1] * 0.30, ' Annealing', ha='left', fontsize=10, alpha=0.7)


    plt.tight_layout()

    # Save both PDF and PNG
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"📄 PDF plot saved to: {pdf_path}")

    plt.close()

def main():
    tokens_per_step = 4194304

    progress_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/emergence_results_corrected/final_results.json")
    checkpoint_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/corrected_checkpoints.json")
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/final_deliverables")

    progressions = print_progression_summary_by_type(tokens_per_step, progress_file, checkpoint_file)

    # Create and save plot    
    plot_checkpoint_progression_by_type(progressions, output_dir, plot_fraction=False)
    plot_checkpoint_progression_by_type(progressions, output_dir, plot_fraction=True)

    # Also save to CSV for easy access
    cloze_df = pd.DataFrame(progressions['cloze'])
    mcqa_df = pd.DataFrame(progressions['mcqa'])

    cloze_output = output_dir / "checkpoint_progression_cloze.csv"
    mcqa_output = output_dir / "checkpoint_progression_mcqa.csv"

    cloze_df.to_csv(cloze_output, index=False)
    mcqa_df.to_csv(mcqa_output, index=False)

    print(f"\n💾 Cloze data saved to: {cloze_output}")
    print(f"💾 MCQA data saved to: {mcqa_output}")


if __name__ == "__main__":  
    main()