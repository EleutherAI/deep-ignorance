#!/usr/bin/env python3
"""
Binary search to find the first checkpoint where each question becomes correct.
"""

import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from evaluate_single_question import SingleQuestionEvaluator

class CheckpointBinarySearch:
    """Binary search to find emergence points of correct answers."""

    def __init__(self, output_dir: str = "/mnt/ssd-1/lucia/deep-ignorance/analysis/emergence_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = SingleQuestionEvaluator()
        self.checkpoints = []
        self.questions = []
        self.results = {}
        self.progress_file = self.output_dir / "progress.json"

    def load_data(self):
        """Load checkpoints and questions."""
        # Load sorted checkpoints
        with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/sorted_checkpoints.json', 'r') as f:
            checkpoint_data = json.load(f)

        self.checkpoints = [(ckpt['step'], ckpt['model_name'], ckpt['revision'])
                           for ckpt in checkpoint_data]

        print(f"Loaded {len(self.checkpoints)} checkpoints")

        # Load questions
        with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/questions_for_binary_search.json', 'r') as f:
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
        step, model_name, revision = self.checkpoints[checkpoint_idx]

        # Load model if needed
        self.evaluator.load_model(model_name, revision)

        # Evaluate question
        return self.evaluator.evaluate_question(question)

    def binary_search_emergence(self, question: Dict) -> Dict:
        """
        Binary search to find first checkpoint where question becomes correct.
        Returns dict with results.
        """
        question_key = self.get_question_key(question)
        print(f"\nSearching for emergence of: {question_key}")
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
            'total_evaluations': 0,
            'search_complete': False,
            'never_correct': False,
            'always_correct': False
        }

        # First, test the final checkpoint (should be correct)
        final_correct = self.test_checkpoint(len(self.checkpoints) - 1, question)
        result['total_evaluations'] += 1

        if not final_correct:
            print("  WARNING: Question not correct at final checkpoint!")
            result['never_correct'] = True
            result['search_complete'] = True
            return result

        # Test the first checkpoint
        first_correct = self.test_checkpoint(0, question)
        result['total_evaluations'] += 1

        if first_correct:
            print("  Question correct at first checkpoint")
            result['always_correct'] = True
            result['emergence_step'] = self.checkpoints[0][0]
            result['emergence_checkpoint_idx'] = 0
            result['emergence_model'] = self.checkpoints[0][1]
            result['emergence_revision'] = self.checkpoints[0][2]
            result['search_complete'] = True
            return result

        # Binary search
        left = 0  # Known incorrect
        right = len(self.checkpoints) - 1  # Known correct

        while right - left > 1:
            mid = (left + right) // 2

            print(f"  Testing checkpoint {mid} (step {self.checkpoints[mid][0]})")
            mid_correct = self.test_checkpoint(mid, question)
            result['total_evaluations'] += 1

            if mid_correct:
                right = mid
            else:
                left = mid

        # Right is the first correct checkpoint
        result['emergence_step'] = self.checkpoints[right][0]
        result['emergence_checkpoint_idx'] = right
        result['emergence_model'] = self.checkpoints[right][1]
        result['emergence_revision'] = self.checkpoints[right][2]
        result['search_complete'] = True

        print(f"  Found emergence at step {result['emergence_step']} (checkpoint {right})")
        print(f"  Total evaluations: {result['total_evaluations']}")

        return result

    def run_search(self, start_idx: int = 0, max_questions: Optional[int] = None):
        """Run binary search for all questions."""
        start_time = time.time()
        processed_count = 0

        for i, question in enumerate(self.questions[start_idx:], start_idx):
            question_key = self.get_question_key(question)

            # Skip if already processed
            if question_key in self.results:
                print(f"Skipping {question_key} (already processed)")
                continue

            # Process question
            result = self.binary_search_emergence(question)
            self.results[question_key] = result
            processed_count += 1

            # Save progress every 10 questions
            if processed_count % 10 == 0:
                self.save_progress()
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count
                remaining = len(self.questions) - len(self.results)
                eta = avg_time * remaining / 3600  # Hours

                print(f"\nProgress: {len(self.results)}/{len(self.questions)} questions")
                print(f"Average time per question: {avg_time:.1f}s")
                print(f"ETA: {eta:.1f} hours")

            # Stop if max reached
            if max_questions and processed_count >= max_questions:
                break

        # Final save
        self.save_progress()

        elapsed = time.time() - start_time
        print(f"\nCompleted {processed_count} questions in {elapsed/3600:.2f} hours")

    def save_final_results(self):
        """Save final results in a clean format."""
        # Filter completed results
        completed_results = {k: v for k, v in self.results.items() if v['search_complete']}

        # Save detailed results
        detailed_file = self.output_dir / "detailed_emergence_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(completed_results, f, indent=2)

        # Create summary
        summary = {
            'total_questions': len(self.questions),
            'completed_searches': len(completed_results),
            'never_correct': sum(1 for r in completed_results.values() if r['never_correct']),
            'always_correct': sum(1 for r in completed_results.values() if r['always_correct']),
            'emerged_during_training': sum(1 for r in completed_results.values()
                                         if not r['never_correct'] and not r['always_correct']),
        }

        # Emergence step distribution
        emergence_steps = [r['emergence_step'] for r in completed_results.values()
                          if r['emergence_step'] is not None]
        if emergence_steps:
            summary['emergence_stats'] = {
                'min_step': min(emergence_steps),
                'max_step': max(emergence_steps),
                'mean_step': sum(emergence_steps) / len(emergence_steps)
            }

        summary_file = self.output_dir / "emergence_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved detailed results to {detailed_file}")
        print(f"Saved summary to {summary_file}")
        print(f"Summary: {summary}")

def main():
    parser = argparse.ArgumentParser(description='Binary search for answer emergence')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Question index to start from')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to process')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with just a few questions')

    args = parser.parse_args()

    # Initialize search
    searcher = CheckpointBinarySearch()
    searcher.load_data()
    searcher.load_progress()

    if args.test_mode:
        print("Running in test mode...")
        args.max_questions = 3

    # Run search
    searcher.run_search(args.start_idx, args.max_questions)

    # Save final results
    searcher.save_final_results()

if __name__ == "__main__":
    main()