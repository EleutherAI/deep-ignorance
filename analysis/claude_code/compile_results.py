#!/usr/bin/env python3
"""
Compile final results from the CORRECTED emergence analysis.
"""

import json
import pandas as pd
from pathlib import Path


def create_emergence_dataset(results: dict) -> pd.DataFrame:
    """Create a pandas DataFrame with emergence data."""
    data = []

    for question_key, result in results.items():
        if result.get('search_complete', False):
            # Determine emergence stage from model name or stage field
            emergence_stage = 'unknown'
            if 'emergence_stage' in result:
                emergence_stage = result['emergence_stage']
            elif 'emergence_model' in result and result['emergence_model']:
                if 'pretraining-stage' in result['emergence_model']:
                    emergence_stage = 'pretraining'
                elif 'deep-ignorance-unfiltered' in result['emergence_model']:
                    emergence_stage = 'annealing'

            data.append({
                'question_key': question_key,
                'task': result['task'],
                'doc_id': result['doc_id'],
                'question': result['question'][:100] + "..." if len(result['question']) > 100 else result['question'],
                'correct_answer': result['correct_answer'],
                'emergence_step': result['emergence_step'],
                'emergence_stage': emergence_stage,
                'total_evaluations': result['total_evaluations'],
                'always_correct': result['always_correct'],
                'never_correct': result['never_correct'],
                'task_category': result['task'].replace('wmdp_bio_robust_', '').replace('wmdp_bio_', '')
            })

    return pd.DataFrame(data)

def save_deliverable_dataset(df: pd.DataFrame, output_dir: Path):
    """Save the final deliverable dataset."""

    # Main dataset
    dataset_file = output_dir / 'question_emergence_dataset.csv'
    df.to_csv(dataset_file, index=False)
    print(f"Main dataset saved to {dataset_file}")

    # Step mapping for each question (the key deliverable)
    step_mapping = {}
    for _, row in df.iterrows():
        step_mapping[row['question_key']] = {
            'task': row['task'],
            'doc_id': row['doc_id'],
            'emergence_step': int(row['emergence_step']) if pd.notna(row['emergence_step']) else None,
            'emergence_stage': row['emergence_stage'],
            'always_correct': bool(row['always_correct']),
            'never_correct': bool(row['never_correct'])
        }

    step_mapping_file = output_dir / 'question_emergence_steps.json'
    with open(step_mapping_file, 'w') as f:
        json.dump(step_mapping, f, indent=2)
    print(f"Step mapping saved to {step_mapping_file}")


def main():
    """Main function to compile all results."""
    results_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/emergence_results_corrected/detailed_emergence_results.json")
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create DataFrame
    df = create_emergence_dataset(results)
    print(f"Created dataset with {len(df)} questions")

    if len(df) == 0:
        print("No completed results found. Analysis may still be running.")
        return

    # Create output directory
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/final_deliverables")
    output_dir.mkdir(exist_ok=True)

    # Save deliverable dataset
    save_deliverable_dataset(df, output_dir)

    print(f"\nAll deliverables saved to {output_dir}")
    print("The main deliverable is: question_emergence_steps.json")

if __name__ == "__main__":
    main()