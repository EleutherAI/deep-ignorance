#!/usr/bin/env python3
"""
Evaluation infrastructure for testing individual questions at specific checkpoints.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import time

class SingleQuestionEvaluator:
    """Evaluates individual questions at specific model checkpoints."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_revision = None

    def load_model(self, model_name: str, revision: str):
        """Load model at specific checkpoint."""
        if self.current_model_name == model_name and self.current_revision == revision:
            # Model already loaded
            return

        print(f"Loading {model_name} at revision {revision}...")
        start_time = time.time()

        # Clear existing model
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.current_model_name = model_name
        self.current_revision = revision

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def format_mcqa_prompt(self, question: str, choices: List[str]) -> str:
        """Format multiple choice question as prompt."""
        prompt = f"The following are multiple choice questions (with answers) about biology.\n\n"
        prompt += f"{question.strip()}\n"
        for i, choice in enumerate(choices):
            prompt += f"{'ABCD'[i]}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def format_cloze_prompt(self, question: str, choices: List[str]) -> str:
        """Format cloze question as prompt."""
        prompt = f"Question: {question.strip()}\nAnswer:"
        return prompt

    def evaluate_question_mcqa(self, question: str, choices: List[str], correct_answer: int) -> bool:
        """
        Evaluate a multiple choice question.
        Returns True if model gets it correct.
        """
        prompt = self.format_mcqa_prompt(question, choices)

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        # Get logits for each answer choice
        choice_tokens = []
        for choice_letter in ["A", "B", "C", "D"]:
            choice_token = self.tokenizer.encode(f" {choice_letter}", add_special_tokens=False)
            if len(choice_token) == 1:
                choice_tokens.append(choice_token[0])
            else:
                # Fallback: use just the letter
                choice_token = self.tokenizer.encode(choice_letter, add_special_tokens=False)
                choice_tokens.append(choice_token[0])

        # Get model output
        with torch.no_grad():
            outputs = self.model(prompt_tokens)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get logits for each choice
        choice_logits = [logits[token_id].item() for token_id in choice_tokens]

        # Predict answer (highest logit)
        predicted_answer = np.argmax(choice_logits)

        return predicted_answer == correct_answer

    def evaluate_question_cloze(self, question: str, choices: List[str], correct_answer: int) -> bool:
        """
        Evaluate a cloze question using perplexity-based scoring.
        Returns True if model gets it correct.
        """
        base_prompt = self.format_cloze_prompt(question, choices)

        choice_scores = []

        for i, choice in enumerate(choices):
            # Create full prompt with this choice
            full_prompt = base_prompt + f" {choice}"

            # Tokenize
            tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

            # Compute log likelihood
            with torch.no_grad():
                outputs = self.model(tokens, labels=tokens)
                loss = outputs.loss.item()

            # Store negative log likelihood (lower is better)
            choice_scores.append(-loss)

        # Predict answer (highest score = lowest loss)
        predicted_answer = np.argmax(choice_scores)

        return predicted_answer == correct_answer

    def evaluate_question(self, question_data: Dict) -> bool:
        """
        Evaluate a single question.
        Returns True if model gets it correct.
        """
        task = question_data['task']
        question = question_data['question']
        choices = question_data['choices']
        correct_answer = question_data['correct_answer']

        if 'cloze' in task.lower():
            return self.evaluate_question_cloze(question, choices, correct_answer)
        else:
            return self.evaluate_question_mcqa(question, choices, correct_answer)

def test_evaluator():
    """Test the evaluator with a sample question."""
    # Load sample questions
    with open('/mnt/ssd-1/lucia/deep-ignorance/analysis/questions_for_binary_search.json', 'r') as f:
        questions = json.load(f)

    # Test with first question
    test_question = questions[0]
    print(f"Testing with question: {test_question['question'][:100]}...")

    evaluator = SingleQuestionEvaluator()

    # Test with final model (should be correct)
    evaluator.load_model("EleutherAI/deep-ignorance-unfiltered", "main")
    result = evaluator.evaluate_question(test_question)
    print(f"Final model result: {result} (expected: True)")

    return evaluator

def main():
    """Test the evaluation infrastructure."""
    evaluator = test_evaluator()
    print("Evaluation infrastructure ready!")

if __name__ == "__main__":
    main()