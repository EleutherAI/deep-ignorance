# Question Emergence Analysis Results

## Overview
This dataset contains the emergence analysis results for 764 questions that the Deep Ignorance unfiltered model answers correctly.

## Key Findings
- **Always Correct**: 218 questions (28.5%)
- **Never Correct**: 142 questions (18.6%)
- **Emerged During Training**: 404 questions (52.9%)

## Training Progression
The model was trained in two phases:
1. **Pretraining**: Steps 1192 → 119209 (500B tokens)
2. **Annealing**: Steps 1192 → 11921 (50B tokens, continues after pretraining)

## Files
- `question_emergence_steps.json`: Step number for each question (main deliverable)
- `question_emergence_dataset.csv`: Full dataset with all metadata
- `emergence_analysis_summary.json`: Detailed analysis results
- `emergence_step_distribution.png`: Histogram of emergence steps
- `emergence_by_stage.png`: Emergence by training stage
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
stage = step_mapping['wmdp_bio_cloze_verified_2']['emergence_stage']
```

## Data Format
Each question has:
- `emergence_step`: First training step where model gets question correct
- `emergence_stage`: Training phase (pretraining/annealing) where emergence occurred
- `always_correct`: True if correct from first checkpoint
- `never_correct`: True if never correct
- Task and question metadata

Generated on: 2025-10-26 09:22:41
