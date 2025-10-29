#!/usr/bin/env python3
"""
Single question evaluator using the official lm_eval harness.
This ensures consistency with the original "correct answers" evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# Import lm_eval modules for direct usage
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

class LMEvalSingleQuestionEvaluator:
    """Evaluates individual questions using the official lm_eval harness."""

    def __init__(self, lm_eval_tasks_path: str = "/mnt/ssd-1/lucia/deep-ignorance/lm_eval_tasks"):
        self.lm_eval_tasks_path = lm_eval_tasks_path
        self.current_model_name = None
        self.current_revision = None


    def evaluate_question(self, question_data: Dict) -> bool:
        """
        Evaluate a single question using lm_eval harness directly.
        Returns True if the model gets the question correct.
        """
        if self.current_model_name is None:
            raise ValueError("No model configured. Call load_model() first.")

        task_name = question_data["task"]

        # Build model args
        model_args = f"pretrained={self.current_model_name}"
        if self.current_revision:
            model_args += f",revision={self.current_revision}"
        model_args += ",dtype=bfloat16"

        try:
            # Create a task manager with our custom task path
            task_manager = TaskManager(include_path=self.lm_eval_tasks_path)

            # Use lm_eval to evaluate on the original task with full dataset
            # Then check if our specific question was answered correctly
            results = simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=[task_name],
                num_fewshot=0,
                batch_size=1,
                device=None,
                log_samples=True,
                task_manager=task_manager
            )

            # Get the samples output
            samples = results.get('samples', {}).get(task_name, [])

            # Find our specific question by doc_id
            target_doc_id = question_data['doc_id']

            for sample in samples:
                if sample.get('doc_id') == target_doc_id:
                    # Check the appropriate metric
                    if "cloze" in task_name.lower():
                        return sample.get("acc_norm", 0.0) == 1.0
                    else:
                        return sample.get("acc", 0.0) == 1.0

            # If we didn't find the question, return False
            print(f"Warning: Question with doc_id {target_doc_id} not found in task {task_name}")
            return False

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return False

    def load_model(self, model_name: str, revision: Optional[str]):
        """Configure the model for evaluation."""
        self.current_model_name = model_name
        self.current_revision = revision
        print(f"Configured model: {model_name} at revision {revision}")

def test_lm_eval_evaluator():
    """Test the LM eval evaluator with a known question."""
    questions_file = "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/questions_for_binary_search.json"

    evaluator = LMEvalSingleQuestionEvaluator()
    evaluator.load_model("EleutherAI/deep-ignorance-unfiltered", None)

    # Load the test questions
    with open(questions_file, 'r') as f:
        questions = json.load(f)

    # Find a cloze question
    test_question = None
    for q in questions:
        if q['task'] == 'wmdp_bio_cloze_verified':
            test_question = q
            break

    if test_question:
        print(f"Testing question: {test_question['question'][:100]}...")
        result = evaluator.evaluate_question(test_question)
        print(f"Result: {result}")
    else:
        print("No test question found")

if __name__ == "__main__":
    test_lm_eval_evaluator()