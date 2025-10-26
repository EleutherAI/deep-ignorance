#!/usr/bin/env python3
"""
Extract correctly answered questions from LM eval harness results.

This script processes JSONL files from lm_eval harness evaluations and extracts
the questions that the model answered correctly. The results are saved in a format
that can be used to later run evaluations on just this subset of questions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load samples from a JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_correct_samples(samples: List[Dict[str, Any]], task_name: str) -> List[Dict[str, Any]]:
    """Extract only the correctly answered samples."""
    correct_samples = []

    for sample in samples:
        # Check if the sample was answered correctly
        # For acc_norm tasks (cloze), check acc_norm field
        # For acc tasks (MCQA), check acc field
        is_correct = False

        if 'acc_norm' in sample:
            is_correct = sample['acc_norm'] == 1.0
        elif 'acc' in sample:
            is_correct = sample['acc'] == 1.0

        if is_correct:
            # Extract relevant information
            correct_sample = {
                'task': task_name,
                'doc_id': sample['doc_id'],
                'doc_hash': sample.get('doc_hash', ''),
                'question': sample['doc']['question'],
                'choices': sample['doc']['choices'],
                'answer': sample['doc']['answer'],
                'target': sample['target'],
            }

            # Add any additional fields from doc
            if 'prompt' in sample['doc']:
                correct_sample['prompt'] = sample['doc']['prompt']
            if 'reasoning' in sample['doc']:
                correct_sample['reasoning'] = sample['doc']['reasoning']

            correct_samples.append(correct_sample)

    return correct_samples


def process_eval_results(results_dir: Path, output_dir: Path):
    """Process all evaluation result files in the directory."""

    # Find all sample JSONL files
    sample_files = list(results_dir.glob("**/samples_*.jsonl"))

    if not sample_files:
        print(f"No sample files found in {results_dir}")
        return

    print(f"Found {len(sample_files)} sample files")

    # Process each file
    all_correct = defaultdict(list)
    stats = {}

    for sample_file in sorted(sample_files):
        # Extract task name from filename
        filename = sample_file.stem
        # Remove 'samples_' prefix and timestamp suffix
        parts = filename.split('_')
        task_name = '_'.join(parts[1:-1])  # Remove 'samples' and timestamp

        print(f"\nProcessing {task_name}...")

        samples = load_samples(sample_file)
        correct_samples = extract_correct_samples(samples, task_name)

        all_correct[task_name] = correct_samples
        stats[task_name] = {
            'total': len(samples),
            'correct': len(correct_samples),
            'accuracy': len(correct_samples) / len(samples) if samples else 0
        }

        print(f"  Total samples: {len(samples)}")
        print(f"  Correct answers: {len(correct_samples)}")
        print(f"  Accuracy: {stats[task_name]['accuracy']:.2%}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all correct answers by task
    for task_name, correct_samples in all_correct.items():
        output_file = output_dir / f"{task_name}_correct.jsonl"
        with open(output_file, 'w') as f:
            for sample in correct_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"\nSaved {len(correct_samples)} correct samples to {output_file}")

    # Save a combined file with all correct answers
    combined_file = output_dir / "all_correct_answers.jsonl"
    with open(combined_file, 'w') as f:
        for task_name, correct_samples in sorted(all_correct.items()):
            for sample in correct_samples:
                f.write(json.dumps(sample) + '\n')
    print(f"\nSaved all correct answers to {combined_file}")

    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")

    # Save doc_ids for each task (for easy filtering later)
    doc_ids_by_task = {}
    for task_name, correct_samples in all_correct.items():
        doc_ids_by_task[task_name] = [s['doc_id'] for s in correct_samples]

    doc_ids_file = output_dir / "correct_doc_ids.json"
    with open(doc_ids_file, 'w') as f:
        json.dump(doc_ids_by_task, f, indent=2)
    print(f"Saved doc IDs to {doc_ids_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_samples = sum(s['total'] for s in stats.values())
    total_correct = sum(s['correct'] for s in stats.values())
    print(f"Total samples: {total_samples}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {total_correct/total_samples:.2%}")
    print("\nBy task group:")

    # Group by task type
    cloze_stats = {k: v for k, v in stats.items() if 'cloze' in k}
    mcqa_stats = {k: v for k, v in stats.items() if 'robust' in k}

    if cloze_stats:
        cloze_total = sum(s['total'] for s in cloze_stats.values())
        cloze_correct = sum(s['correct'] for s in cloze_stats.values())
        print(f"  WMDP Bio Cloze Verified: {cloze_correct}/{cloze_total} ({cloze_correct/cloze_total:.2%})")

    if mcqa_stats:
        mcqa_total = sum(s['total'] for s in mcqa_stats.values())
        mcqa_correct = sum(s['correct'] for s in mcqa_stats.values())
        print(f"  WMDP Bio MCQA Robust: {mcqa_correct}/{mcqa_total} ({mcqa_correct/mcqa_total:.2%})")


def main():
    parser = argparse.ArgumentParser(description='Extract correctly answered questions from LM eval results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted correct answers')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    process_eval_results(results_dir, output_dir)


if __name__ == '__main__':
    main()
