#!/usr/bin/env python3
"""
Compile final results from the emergence analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

def load_results() -> Dict:
    """Load the emergence analysis results."""
    results_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results/detailed_emergence_results.json")

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return {}

    with open(results_file, 'r') as f:
        return json.load(f)

def create_emergence_dataset(results: Dict) -> pd.DataFrame:
    """Create a pandas DataFrame with emergence data."""
    data = []

    for question_key, result in results.items():
        if result['search_complete']:
            data.append({
                'question_key': question_key,
                'task': result['task'],
                'doc_id': result['doc_id'],
                'question': result['question'][:100] + "..." if len(result['question']) > 100 else result['question'],
                'correct_answer': result['correct_answer'],
                'emergence_step': result['emergence_step'],
                'emergence_model': 'pretraining' if 'pretraining-stage' in result.get('emergence_model', '') else 'annealing',
                'total_evaluations': result['total_evaluations'],
                'always_correct': result['always_correct'],
                'never_correct': result['never_correct'],
                'task_category': result['task'].replace('wmdp_bio_robust_', '').replace('wmdp_bio_', '')
            })

    return pd.DataFrame(data)

def analyze_emergence_patterns(df: pd.DataFrame) -> Dict:
    """Analyze patterns in the emergence data."""
    analysis = {}

    # Basic statistics
    analysis['total_questions'] = len(df)
    analysis['always_correct'] = df['always_correct'].sum()
    analysis['never_correct'] = df['never_correct'].sum()
    analysis['emerged_during_training'] = len(df[~df['always_correct'] & ~df['never_correct']])

    # Emergence step statistics
    emerged_questions = df[~df['always_correct'] & ~df['never_correct']]
    if len(emerged_questions) > 0:
        analysis['emergence_steps'] = {
            'min': emerged_questions['emergence_step'].min(),
            'max': emerged_questions['emergence_step'].max(),
            'mean': emerged_questions['emergence_step'].mean(),
            'median': emerged_questions['emergence_step'].median(),
            'std': emerged_questions['emergence_step'].std()
        }

    # By task category
    analysis['by_task'] = {}
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        analysis['by_task'][task] = {
            'total': len(task_data),
            'always_correct': task_data['always_correct'].sum(),
            'never_correct': task_data['never_correct'].sum(),
            'mean_emergence_step': task_data[~task_data['always_correct'] & ~task_data['never_correct']]['emergence_step'].mean()
        }

    # By model type
    analysis['by_model'] = {}
    for model in df['emergence_model'].unique():
        model_data = df[df['emergence_model'] == model]
        analysis['by_model'][model] = {
            'total': len(model_data),
            'always_correct': model_data['always_correct'].sum(),
            'never_correct': model_data['never_correct'].sum()
        }

    return analysis

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of the emergence patterns."""
    plt.style.use('default')

    # Emergence step distribution
    emerged_questions = df[~df['always_correct'] & ~df['never_correct']]
    if len(emerged_questions) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(emerged_questions['emergence_step'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Emergence Step')
        plt.ylabel('Number of Questions')
        plt.title('Distribution of Question Emergence Steps')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'emergence_step_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # By task category
    task_summary = df.groupby('task_category').agg({
        'always_correct': 'sum',
        'never_correct': 'sum',
        'emergence_step': ['count', 'mean']
    }).round(2)

    plt.figure(figsize=(14, 8))
    task_counts = df['task_category'].value_counts()
    plt.bar(task_counts.index, task_counts.values)
    plt.xlabel('Task Category')
    plt.ylabel('Number of Questions')
    plt.title('Questions by Task Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'questions_by_task.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")

def save_deliverable_dataset(df: pd.DataFrame, analysis: Dict, output_dir: Path):
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
            'emergence_step': row['emergence_step'],
            'always_correct': row['always_correct'],
            'never_correct': row['never_correct']
        }

    step_mapping_file = output_dir / 'question_emergence_steps.json'
    with open(step_mapping_file, 'w') as f:
        json.dump(step_mapping, f, indent=2)
    print(f"Step mapping saved to {step_mapping_file}")

    # Analysis summary
    analysis_file = output_dir / 'emergence_analysis_summary.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis summary saved to {analysis_file}")

    # Create README for deliverable
    readme_content = f"""# Question Emergence Analysis Results

## Overview
This dataset contains the emergence analysis results for {analysis['total_questions']} questions that the Deep Ignorance unfiltered model answers correctly.

## Key Findings
- **Always Correct**: {analysis['always_correct']} questions ({analysis['always_correct']/analysis['total_questions']*100:.1f}%)
- **Never Correct**: {analysis['never_correct']} questions ({analysis['never_correct']/analysis['total_questions']*100:.1f}%)
- **Emerged During Training**: {analysis['emerged_during_training']} questions ({analysis['emerged_during_training']/analysis['total_questions']*100:.1f}%)

## Files
- `question_emergence_steps.json`: Step number for each question (main deliverable)
- `question_emergence_dataset.csv`: Full dataset with all metadata
- `emergence_analysis_summary.json`: Detailed analysis results
- `emergence_step_distribution.png`: Histogram of emergence steps
- `questions_by_task.png`: Distribution by task category

## Usage
```python
import json
import pandas as pd

# Load step mapping
with open('question_emergence_steps.json', 'r') as f:
    step_mapping = json.load(f)

# Load full dataset
df = pd.read_csv('question_emergence_dataset.csv')

# Get emergence step for a specific question
step = step_mapping['wmdp_bio_cloze_verified_2']['emergence_step']
```

## Data Format
Each question has:
- `emergence_step`: First training step where model gets question correct
- `always_correct`: True if correct from first checkpoint
- `never_correct`: True if never correct (should be rare)
- Task and question metadata

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"README saved to {readme_file}")

def main():
    """Main function to compile all results."""
    print("Compiling emergence analysis results...")

    # Load results
    results = load_results()
    if not results:
        print("No results found. Make sure the analysis has completed.")
        return

    print(f"Loaded {len(results)} questions")

    # Create DataFrame
    df = create_emergence_dataset(results)
    print(f"Created dataset with {len(df)} questions")

    # Analyze patterns
    analysis = analyze_emergence_patterns(df)
    print("Analysis complete:")
    print(f"- Total questions: {analysis['total_questions']}")
    print(f"- Always correct: {analysis['always_correct']}")
    print(f"- Never correct: {analysis['never_correct']}")
    print(f"- Emerged during training: {analysis['emerged_during_training']}")

    # Create output directory
    output_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/final_deliverables")
    output_dir.mkdir(exist_ok=True)

    # Create visualizations
    create_visualizations(df, output_dir)

    # Save deliverable dataset
    save_deliverable_dataset(df, analysis, output_dir)

    print(f"\nAll deliverables saved to {output_dir}")
    print("The main deliverable is: question_emergence_steps.json")

if __name__ == "__main__":
    main()