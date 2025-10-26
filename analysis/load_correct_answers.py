#!/usr/bin/env python3
"""
Load and analyze the correct answers dataset.
"""

import json
from typing import List, Dict, Tuple
from collections import defaultdict

def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct answers from JSONL file."""
    answers = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers

def analyze_correct_answers(answers: List[Dict]):
    """Analyze the correct answers dataset."""
    print(f"Total correct answers: {len(answers)}")

    # Group by task
    by_task = defaultdict(list)
    for answer in answers:
        by_task[answer['task']].append(answer)

    print("\nBy task:")
    for task, task_answers in by_task.items():
        print(f"  {task}: {len(task_answers)}")

    # Analyze answer distribution
    answer_dist = defaultdict(int)
    for answer in answers:
        answer_dist[answer['answer']] += 1

    print("\nAnswer distribution:")
    for ans, count in sorted(answer_dist.items()):
        print(f"  Answer {ans}: {count} ({count/len(answers)*100:.1f}%)")

    return by_task

def create_question_index(answers: List[Dict]) -> Dict[Tuple[str, int], Dict]:
    """Create an index of questions by (task, doc_id)."""
    question_index = {}
    for answer in answers:
        key = (answer['task'], answer['doc_id'])
        question_index[key] = answer
    return question_index

def save_analysis_data(answers: List[Dict], by_task: Dict[str, List[Dict]]):
    """Save processed data for binary search."""
    # Create a simplified format for binary search
    questions_for_search = []
    for answer in answers:
        question_data = {
            'task': answer['task'],
            'doc_id': answer['doc_id'],
            'doc_hash': answer['doc_hash'],
            'question': answer['question'],
            'choices': answer['choices'],
            'correct_answer': answer['answer'],
            'target': answer['target']
        }
        questions_for_search.append(question_data)

    # Save to file
    output_file = '/mnt/ssd-1/lucia/deep-ignorance/analysis/questions_for_binary_search.json'
    with open(output_file, 'w') as f:
        json.dump(questions_for_search, f, indent=2)

    print(f"\nSaved {len(questions_for_search)} questions for binary search to {output_file}")

    # Also save summary statistics
    summary = {
        'total_questions': len(answers),
        'by_task': {task: len(task_answers) for task, task_answers in by_task.items()},
        'answer_distribution': {
            str(i): sum(1 for a in answers if a['answer'] == i) for i in range(4)
        }
    }

    summary_file = '/mnt/ssd-1/lucia/deep-ignorance/analysis/correct_answers_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_file}")

def main():
    # Load correct answers
    file_path = '/mnt/ssd-1/lucia/deep-ignorance/analysis/unfiltered_correct_answers/all_correct_answers.jsonl'
    answers = load_correct_answers(file_path)

    # Analyze
    by_task = analyze_correct_answers(answers)

    # Create question index
    question_index = create_question_index(answers)
    print(f"\nCreated question index with {len(question_index)} unique questions")

    # Save processed data
    save_analysis_data(answers, by_task)

if __name__ == "__main__":
    main()