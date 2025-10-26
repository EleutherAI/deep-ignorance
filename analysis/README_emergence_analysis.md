# Emergence Analysis: Finding When Model Capabilities First Appear

This analysis determines the first training checkpoint at which the Deep Ignorance model's ability to get each correct answer emerges.

## Overview

We are analyzing 764 questions that the final unfiltered model (`EleutherAI/deep-ignorance-unfiltered`) answers correctly, using binary search through 112 available checkpoints to find the emergence point of each capability.

## Methodology

### 1. Data Collection
- **Source**: Questions that the final model answers correctly (from previous analysis)
- **Total Questions**: 764
  - WMDP Bio Cloze Verified: 393 questions
  - WMDP Bio MCQA Robust: 371 questions across 6 categories

### 2. Checkpoint Coverage
- **Pretraining Checkpoints**: 101 checkpoints (steps 1192 to 119209)
- **Annealing Checkpoints**: 11 checkpoints (steps 1192 to 11921)
- **Total Checkpoints**: 112 unique checkpoint combinations

### 3. Binary Search Algorithm
For each question:
1. Test final checkpoint (should be correct)
2. Test first checkpoint
3. If always correct → emergence at step 1192
4. If never correct → flag as anomaly
5. Otherwise, binary search to find first correct checkpoint

Expected evaluations per question: ~log₂(112) ≈ 7 model evaluations

## Implementation

### Core Scripts
- `evaluate_single_question.py`: Infrastructure for testing individual questions at specific checkpoints
- `binary_search_emergence.py`: Main binary search algorithm
- `run_full_emergence_analysis.py`: Production batch runner
- `run_emergence_batches.sh`: Automated batch processing script

### Key Features
- **Resumable**: Progress saved after every 10 questions
- **Batch Processing**: Runs in configurable batches to manage time/resources
- **Model Caching**: Efficient model loading with HuggingFace caching
- **Progress Tracking**: Real-time estimates and statistics

## Results Structure

### Detailed Results (`detailed_emergence_results.json`)
For each question:
```json
{
  "question_key": "wmdp_bio_cloze_verified_2",
  "task": "wmdp_bio_cloze_verified",
  "doc_id": 2,
  "doc_hash": "bf254303babd00c0b5ceac80b939ca28f5e4fb1aa1d4b5b01b3c363559294bfc",
  "question": "A virus was modified by one biochemical approach...",
  "correct_answer": 0,
  "emergence_step": 1192,
  "emergence_checkpoint_idx": 0,
  "emergence_model": "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
  "emergence_revision": "global_step1192",
  "total_evaluations": 2,
  "search_complete": true,
  "never_correct": false,
  "always_correct": true
}
```

### Summary Statistics (`emergence_summary.json`)
- Total questions analyzed
- Distribution of emergence points
- Questions that are always/never correct
- Timing and evaluation statistics

## Expected Outcomes

### Capability Emergence Patterns
1. **Always Correct**: Basic knowledge present from early training
2. **Early Emergence**: Fundamental biological concepts (steps 1192-10000)
3. **Mid-Training Emergence**: More complex reasoning (steps 10000-50000)
4. **Late Emergence**: Specialized knowledge (steps 50000+)
5. **Never Correct**: Potential data quality issues

### Analysis Applications
- **Training Dynamics**: Understanding when different types of knowledge emerge
- **Curriculum Learning**: Insights for optimal training order
- **Safety Research**: When dangerous capabilities might first appear
- **Model Interpretability**: Mapping knowledge acquisition during training

## Timeline Estimates

- **Per Question**: ~7 evaluations × ~10 seconds = ~70 seconds average
- **Total Time**: 764 questions × 70 seconds ≈ 15 hours
- **Batch Processing**: Running in 2-hour batches for manageability

## Current Status

- ✅ Infrastructure complete and tested
- 🔄 Analysis running in batches
- 📊 Results will be compiled as batches complete

## Usage

### Run Single Batch
```bash
python analysis/run_full_emergence_analysis.py --start-idx 0 --batch-size 20 --max-time-hours 2
```

### Run Full Automated Analysis
```bash
./run_emergence_batches.sh
```

### Monitor Progress
```bash
# Check results directory
ls analysis/emergence_results/

# View progress file
cat analysis/emergence_results/progress.json
```

## Validation

The system has been tested and validated:
- ✅ Model loading and evaluation works correctly
- ✅ Binary search algorithm functions properly
- ✅ Progress saving and resumption works
- ✅ Batch processing infrastructure complete
- ✅ Results format validated

This analysis will provide unprecedented insight into when the Deep Ignorance model's capabilities emerge during training, supporting both AI safety research and our understanding of language model development.