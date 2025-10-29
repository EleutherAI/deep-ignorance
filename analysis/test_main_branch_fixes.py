#!/usr/bin/env python3
"""
Test the 137 'never_correct' questions on the main branch to see if they're actually correct there.
"""

import json
import sys
from pathlib import Path
from evaluate_single_question import SingleQuestionEvaluator

def test_never_correct_on_main():
    """Test never_correct questions on main branch."""

    # Load emergence data
    progress_file = Path("results/emergence_results_corrected/progress.json")
    with open(progress_file, 'r') as f:
        emergence_data = json.load(f)

    # Load original questions
    with open('results/questions_for_binary_search.json', 'r') as f:
        original_questions = json.load(f)

    # Create lookup for original questions
    question_lookup = {}
    for q in original_questions:
        key = f"{q['task']}_{q['doc_id']}"
        question_lookup[key] = q

    # Find never_correct cloze questions
    never_correct_cloze = []
    for question_key, result in emergence_data.items():
        if (result.get('task', '').startswith('wmdp_bio_cloze') and
            result.get('never_correct', False)):
            never_correct_cloze.append(question_key)

    print(f"=== TESTING {len(never_correct_cloze)} NEVER_CORRECT CLOZE QUESTIONS ON MAIN BRANCH ===")
    print()

    # Load evaluator
    evaluator = SingleQuestionEvaluator()
    evaluator.load_model('EleutherAI/deep-ignorance-unfiltered', None)  # None = main branch

    # Test each question
    corrected_on_main = 0
    still_incorrect = 0
    errors = 0

    results = {}

    for i, question_key in enumerate(never_correct_cloze):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(never_correct_cloze)}")

        # Get the original question data
        original_question = question_lookup.get(question_key)
        if not original_question:
            print(f"Warning: Could not find original question for {question_key}")
            errors += 1
            continue

        try:
            # Test on main branch
            is_correct = evaluator.evaluate_question(original_question)
            results[question_key] = is_correct

            if is_correct:
                corrected_on_main += 1
            else:
                still_incorrect += 1

        except Exception as e:
            print(f"Error testing {question_key}: {e}")
            errors += 1

    print(f"\\n=== RESULTS ===")
    print(f"Total questions tested: {len(never_correct_cloze)}")
    print(f"Now correct on main branch: {corrected_on_main}")
    print(f"Still incorrect: {still_incorrect}")
    print(f"Errors: {errors}")
    print()

    if corrected_on_main > 0:
        print(f"🎯 SUCCESS! {corrected_on_main} questions are now correct on main branch!")
        print("This explains the discrepancy between 256 and 393.")

        # Calculate expected final numbers
        current_cloze_correct = 256  # From our previous analysis
        expected_final_cloze = current_cloze_correct + corrected_on_main
        print(f"Expected final cloze count: {current_cloze_correct} + {corrected_on_main} = {expected_final_cloze}")

        # Save the corrected results
        corrections_file = Path("results/main_branch_corrections.json")
        with open(corrections_file, 'w') as f:
            json.dump({
                'corrected_on_main': corrected_on_main,
                'still_incorrect': still_incorrect,
                'detailed_results': results,
                'expected_final_cloze_count': expected_final_cloze
            }, f, indent=2)
        print(f"\\nSaved detailed results to {corrections_file}")
    else:
        print("No questions were corrected on main branch. The discrepancy remains unexplained.")

    return results

if __name__ == "__main__":
    test_never_correct_on_main()