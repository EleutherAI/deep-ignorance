#!/usr/bin/env python3
"""
Corrected binary search using proper temporal checkpoint ordering.
"""

import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from evaluate_single_question import SingleQuestionEvaluator

class CorrectedCheckpointBinarySearch:
    """Binary search with correct temporal checkpoint ordering."""

    def __init__(self, output_dir: str = "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/emergence_results_corrected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = SingleQuestionEvaluator()
        self.checkpoints = []  # Will be loaded with correct ordering
        self.questions = []
        self.results = {}
        self.progress_file = self.output_dir / "progress.json"

    def load_data(self):
        """Load checkpoints with CORRECT temporal ordering and questions."""
        # Load corrected checkpoints
        with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/results/corrected_checkpoints.json', 'r') as f:
            checkpoint_data = json.load(f)

        # Convert to tuple format for binary search
        self.checkpoints = []
        for ckpt in checkpoint_data:
            self.checkpoints.append((
                ckpt['step'],
                ckpt['model_name'],
                ckpt['revision'],
                ckpt['stage'],
                ckpt['temporal_order']
            ))

        print(f"Loaded {len(self.checkpoints)} checkpoints in CORRECT temporal order")
        print(f"First checkpoint: {self.checkpoints[0][3]} step {self.checkpoints[0][0]}")
        print(f"Last checkpoint: {self.checkpoints[-1][3]} step {self.checkpoints[-1][0]}")

        # Load questions
        with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/results/questions_for_binary_search.json', 'r') as f:
            self.questions = json.load(f)

        print(f"Loaded {len(self.questions)} questions")

    def load_progress(self):
        """Load existing progress if available."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded progress: {len(self.results)} questions already processed")
        else:
            self.results = {}

    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_question_key(self, question: Dict) -> str:
        """Generate unique key for a question."""
        return f"{question['task']}_{question['doc_id']}"

    def test_checkpoint(self, checkpoint_idx: int, question: Dict) -> bool:
        """Test if model at checkpoint gets question correct."""
        step, model_name, revision, stage, temporal_order = self.checkpoints[checkpoint_idx]

        print(f"    Testing checkpoint {checkpoint_idx}: {stage} step {step}")

        # Load model if needed
        self.evaluator.load_model(model_name, revision)

        # Evaluate question
        return self.evaluator.evaluate_question(question)

    def binary_search_emergence(self, question: Dict) -> Dict:
        """
        Binary search to find first checkpoint where question becomes correct.
        Uses CORRECT temporal ordering.
        """
        question_key = self.get_question_key(question)
        print(f"\n🔍 Searching for emergence of: {question_key}")
        print(f"Question: {question['question'][:100]}...")

        result = {
            'question_key': question_key,
            'task': question['task'],
            'doc_id': question['doc_id'],
            'doc_hash': question['doc_hash'],
            'question': question['question'],
            'correct_answer': question['correct_answer'],
            'emergence_step': None,
            'emergence_checkpoint_idx': None,
            'emergence_model': None,
            'emergence_revision': None,
            'emergence_stage': None,
            'total_evaluations': 0,
            'search_complete': False,
            'never_correct': False,
            'always_correct': False
        }

        # Test the FINAL checkpoint (should be correct)
        final_idx = len(self.checkpoints) - 1
        final_correct = self.test_checkpoint(final_idx, question)
        result['total_evaluations'] += 1

        if not final_correct:
            print("  ❌ WARNING: Question not correct at final checkpoint!")
            result['never_correct'] = True
            result['search_complete'] = True
            return result

        # Test the FIRST checkpoint
        first_correct = self.test_checkpoint(0, question)
        result['total_evaluations'] += 1

        if first_correct:
            print("  ✅ Question correct at first checkpoint")
            result['always_correct'] = True
            step, model_name, revision, stage, temporal_order = self.checkpoints[0]
            result['emergence_step'] = step
            result['emergence_checkpoint_idx'] = 0
            result['emergence_model'] = model_name
            result['emergence_revision'] = revision
            result['emergence_stage'] = stage
            result['search_complete'] = True
            return result

        # Binary search between first (incorrect) and final (correct)
        left = 0  # Known incorrect
        right = final_idx  # Known correct

        print(f"  🔍 Binary search between checkpoints {left} and {right}")

        while right - left > 1:
            mid = (left + right) // 2

            mid_correct = self.test_checkpoint(mid, question)
            result['total_evaluations'] += 1

            if mid_correct:
                right = mid
                print(f"    ✅ Checkpoint {mid} correct, searching earlier...")
            else:
                left = mid
                print(f"    ❌ Checkpoint {mid} incorrect, searching later...")

        # Right is the first correct checkpoint
        step, model_name, revision, stage, temporal_order = self.checkpoints[right]
        result['emergence_step'] = step
        result['emergence_checkpoint_idx'] = right
        result['emergence_model'] = model_name
        result['emergence_revision'] = revision
        result['emergence_stage'] = stage
        result['search_complete'] = True

        print(f"  🎯 Found emergence at {stage} step {step} (checkpoint {right})")
        print(f"  📊 Total evaluations: {result['total_evaluations']}")

        return result

    def run_search(self, start_idx: int = 0, max_questions: Optional[int] = None):
        """Run binary search for questions."""
        start_time = time.time()
        processed_count = 0

        for i, question in enumerate(self.questions[start_idx:], start_idx):
            question_key = self.get_question_key(question)

            # Skip if already processed
            if question_key in self.results:
                print(f"⏭️  Skipping {question_key} (already processed)")
                continue

            # Process question
            result = self.binary_search_emergence(question)
            self.results[question_key] = result
            processed_count += 1

            # Save progress every 5 questions
            if processed_count % 5 == 0:
                self.save_progress()
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count
                remaining = len(self.questions) - len(self.results)
                eta = avg_time * remaining / 3600  # Hours

                print(f"\n📈 Progress: {len(self.results)}/{len(self.questions)} questions")
                print(f"⏱️  Average time per question: {avg_time:.1f}s")
                print(f"🕐 ETA: {eta:.1f} hours")

            # Stop if max reached
            if max_questions and processed_count >= max_questions:
                break

        # Final save
        self.save_progress()

        elapsed = time.time() - start_time
        print(f"\n✅ Completed {processed_count} questions in {elapsed/3600:.2f} hours")

def main():
    parser = argparse.ArgumentParser(description='Corrected binary search for answer emergence')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Question index to start from')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to process')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with just a few questions')

    args = parser.parse_args()

    # Initialize search with corrected ordering
    searcher = CorrectedCheckpointBinarySearch()
    searcher.load_data()
    searcher.load_progress()

    if args.test_mode:
        print("🧪 Running in test mode...")
        args.max_questions = 3

    # Run search
    searcher.run_search(args.start_idx, args.max_questions)

if __name__ == "__main__":
    main()