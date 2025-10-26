#!/usr/bin/env python3
"""
Production script to run the full emergence analysis in batches.
"""

import argparse
import time
import json
from pathlib import Path
from binary_search_emergence import CheckpointBinarySearch

def estimate_time_remaining(searcher, batch_size=50):
    """Estimate time remaining for the full analysis."""
    # Sample timing for a few questions to get accurate estimates
    completed = len(searcher.results)
    remaining = len(searcher.questions) - completed

    if completed == 0:
        print("No completed questions yet - running initial batch to estimate timing...")
        return None

    # Use completed results to estimate timing
    total_evaluations = sum(r.get('total_evaluations', 0) for r in searcher.results.values())
    avg_evaluations_per_question = total_evaluations / completed if completed > 0 else 7

    # Estimate time per evaluation (including model loading)
    # This will vary but we can use rough estimates:
    # - First load of a checkpoint: ~60 seconds
    # - Subsequent loads: ~4 seconds
    # - Evaluation itself: ~1 second
    avg_time_per_evaluation = 10  # Conservative estimate

    estimated_remaining_time = remaining * avg_evaluations_per_question * avg_time_per_evaluation

    print(f"\nTiming Estimates:")
    print(f"Completed questions: {completed}")
    print(f"Remaining questions: {remaining}")
    print(f"Average evaluations per question: {avg_evaluations_per_question:.1f}")
    print(f"Estimated remaining time: {estimated_remaining_time/3600:.1f} hours")

    return estimated_remaining_time

def run_batch(start_idx=0, batch_size=50, max_time_hours=4):
    """Run a batch of questions with time limit."""
    searcher = CheckpointBinarySearch()
    searcher.load_data()
    searcher.load_progress()

    # Estimate timing
    estimate_time_remaining(searcher)

    # Calculate max questions for this batch
    max_time_seconds = max_time_hours * 3600

    print(f"\nStarting batch from question {start_idx}")
    print(f"Batch size: {batch_size}")
    print(f"Max time: {max_time_hours} hours")

    start_time = time.time()

    # Run the search
    searcher.run_search(start_idx=start_idx, max_questions=batch_size)

    elapsed = time.time() - start_time
    print(f"\nBatch completed in {elapsed/3600:.2f} hours")

    # Save final results
    searcher.save_final_results()

    return len(searcher.results)

def create_run_script():
    """Create a shell script to run batches automatically."""
    script_content = """#!/bin/bash
# Automated batch processing for emergence analysis

BATCH_SIZE=20
MAX_TIME_HOURS=2

echo "Starting automated emergence analysis..."
echo "Batch size: $BATCH_SIZE questions"
echo "Max time per batch: $MAX_TIME_HOURS hours"

# Function to run a batch
run_batch() {
    local start_idx=$1
    echo ""
    echo "=================================="
    echo "Starting batch from question $start_idx"
    echo "=================================="

    python analysis/run_full_emergence_analysis.py \\
        --start-idx $start_idx \\
        --batch-size $BATCH_SIZE \\
        --max-time-hours $MAX_TIME_HOURS

    if [ $? -ne 0 ]; then
        echo "Batch failed! Stopping..."
        exit 1
    fi
}

# Run batches
for start_idx in $(seq 0 $BATCH_SIZE 763); do
    run_batch $start_idx

    # Check if we've completed all questions
    if [ $start_idx -gt 700 ]; then
        echo "Approaching end of questions..."
    fi
done

echo ""
echo "All batches completed!"
echo "Check results in analysis/emergence_results/"
"""

    script_path = Path("/mnt/ssd-1/lucia/deep-ignorance/run_emergence_batches.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)  # Make executable
    print(f"Created batch script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description='Run full emergence analysis in batches')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Question index to start from')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of questions per batch')
    parser.add_argument('--max-time-hours', type=float, default=4.0,
                       help='Maximum time per batch in hours')
    parser.add_argument('--create-script', action='store_true',
                       help='Create automated batch script')

    args = parser.parse_args()

    if args.create_script:
        create_run_script()
        return

    completed = run_batch(args.start_idx, args.batch_size, args.max_time_hours)
    print(f"\nBatch completed. Total questions processed: {completed}")

if __name__ == "__main__":
    main()