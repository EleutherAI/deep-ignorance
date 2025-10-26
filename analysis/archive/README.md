# Archive Directory

This directory contains outdated analysis files that were superseded during development due to critical fixes in temporal ordering and improved implementations.

## Archived Files

### Binary Search Implementation
- **`binary_search_emergence.py`**: Original binary search implementation with incorrect temporal ordering
  - **Issue**: Mixed pretraining and annealing checkpoints incorrectly
  - **Replaced by**: `../corrected_binary_search.py`

### Progression Analysis Implementations
- **`simple_progression.py`**: Early simple progression analysis
  - **Replaced by**: `../get_checkpoint_progression.py`

- **`corrected_progression.py`**: Intermediate corrected progression analysis
  - **Replaced by**: `../get_checkpoint_progression.py`

- **`checkpoint_progression.py`**: Another progression analysis implementation
  - **Replaced by**: `../get_checkpoint_progression.py` and `../get_checkpoint_progression_by_type.py`

## Why These Were Replaced

The main issue was **temporal ordering**: the original implementations incorrectly assumed that step numbers could be used directly to order checkpoints. However, the training actually follows this sequence:

1. **Pretraining**: steps 1192 → 119209 (checkpoints 0-100)
2. **Annealing**: steps 1192 → 11921 (checkpoints 101-111) - note step numbers restart!

This meant annealing checkpoints were being placed before pretraining checkpoints when sorting by step number, leading to completely incorrect emergence analysis.

## Current Working Files

- `../corrected_binary_search.py`: Fixed binary search with proper temporal ordering
- `../get_checkpoint_progression.py`: Main progression analysis function
- `../get_checkpoint_progression_by_type.py`: Progression analysis split by question type (cloze vs MCQA)