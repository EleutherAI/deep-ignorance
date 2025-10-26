# Correctly Answered Questions - EleutherAI/deep-ignorance-unfiltered

This directory contains the WMDP bio questions that the `EleutherAI/deep-ignorance-unfiltered` model answered correctly.

## Evaluation Results Summary

**Model**: EleutherAI/deep-ignorance-unfiltered
**Date**: 2025-10-25
**Total Questions**: 1,944
**Correctly Answered**: 764 (39.30%)

### By Task Group

| Task Group | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| WMDP Bio Cloze Verified | 393 | 1,076 | 36.52% |
| WMDP Bio MCQA Robust | 371 | 868 | 42.74% |

### By Category (MCQA Robust)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Reverse Genetics and Easy Editing | 95 | 186 | 51.08% |
| Viral Vector Research | 155 | 341 | 45.45% |
| Enhanced Potential Pandemic Pathogens | 38 | 102 | 37.25% |
| Expanding Access to Threat Vectors | 8 | 21 | 38.10% |
| Bioweapons and Bioterrorism | 66 | 190 | 34.74% |
| Dual Use Virology | 9 | 28 | 32.14% |

## Files in this Directory

### Question Data Files

- **`all_correct_answers.jsonl`** (740 KB): All 764 correctly answered questions in a single JSONL file
- **`wmdp_bio_cloze_verified_correct.jsonl`** (531 KB): 393 correctly answered cloze questions
- **`wmdp_bio_robust_<category>_correct.jsonl`**: Correctly answered questions for each MCQA category

### Metadata Files

- **`statistics.json`**: Detailed statistics for each task (total, correct, accuracy)
- **`correct_doc_ids.json`**: Document IDs of correctly answered questions, organized by task
- **`README.md`**: This file

## Data Format

Each line in the JSONL files contains one correctly answered question with the following structure:

```json
{
  "task": "wmdp_bio_cloze_verified",
  "doc_id": 2,
  "doc_hash": "bf254303babd00c0b5ceac80b939ca28f5e4fb1aa1d4b5b01b3c363559294bfc",
  "question": "Question text here...",
  "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
  "answer": 0,
  "target": "0",
  "prompt": "Full prompt with question and choices...",
  "reasoning": "Explanation of why this is a valid cloze question..."
}
```

### Fields

- **task**: Name of the evaluation task
- **doc_id**: Original document ID from the evaluation dataset
- **doc_hash**: Hash of the document for verification
- **question**: The question text
- **choices**: List of 4 multiple choice options
- **answer**: Index of the correct answer (0-3)
- **target**: String representation of the correct answer index
- **prompt**: Full formatted prompt (cloze tasks only)
- **reasoning**: Explanation of question validity (cloze tasks only)

## Usage

### Loading Correctly Answered Questions

```python
import json

# Load all correct answers
correct_answers = []
with open('all_correct_answers.jsonl', 'r') as f:
    for line in f:
        correct_answers.append(json.loads(line))

print(f"Loaded {len(correct_answers)} correctly answered questions")
```

### Loading Doc IDs for Filtering

```python
import json

# Load doc IDs by task
with open('correct_doc_ids.json', 'r') as f:
    correct_ids = json.load(f)

# Get IDs for a specific task
cloze_correct_ids = correct_ids['wmdp_bio_cloze_verified']
print(f"Cloze questions answered correctly: {len(cloze_correct_ids)}")
```

### Filtering Evaluation Dataset

To run evaluations on only the correctly answered questions, you can:

1. Use the `doc_id` values to filter the original WMDP dataset
2. Use the `doc_hash` values to verify question identity
3. Create a custom evaluation task that includes only these questions

## Reproduction

To reproduce these results:

```bash
# Run evaluation with logging
lm_eval --model hf \
  --model_args pretrained=EleutherAI/deep-ignorance-unfiltered,dtype=bfloat16,parallelize=True \
  --tasks wmdp_bio_robust,wmdp_bio_cloze_verified \
  --batch_size 64 \
  --include_path ./lm_eval_tasks/ \
  --log_samples \
  --output_path ./eval_results/unfiltered_wmdp_bio

# Extract correctly answered questions
python extract_correct_answers.py \
  --results_dir ./eval_results/unfiltered_wmdp_bio \
  --output_dir ./analysis/unfiltered_correct_answers
```

## Notes

- WMDP (Weapons of Mass Destruction Proxy) uses proxy knowledge that is adjacent to but not actually harmful information
- This dataset is specifically designed to be safe for AI safety research
- The questions test knowledge that often appears alongside biorisk information but is not itself harmful
- These results can be used to understand what knowledge the unfiltered model retains, which helps inform filtering strategies

## Related Files

- Original evaluation results: `../eval_results/unfiltered_wmdp_bio/`
- Extraction script: `../../extract_correct_answers.py`
- Evaluation task definitions: `../../lm_eval_tasks/`
