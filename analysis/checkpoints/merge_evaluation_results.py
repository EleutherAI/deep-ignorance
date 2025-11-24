#!/usr/bin/env python3
"""
Extract correctly answered questions from LM eval harness results.

This script processes JSONL files from lm_eval harness evaluations and extracts
the questions that the model answered correctly. The results are saved in a format
that can be used to later run evaluations on just this subset of questions.
"""

import json
import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def extract_correct_samples(
    results: dict[str, Any], const_sample_data: dict
) -> list[dict]:
    """Extract only the correctly answered samples."""
    samples_data = []

    for task_name, task_data in results["samples"].items():
        for sample in task_data:
            # Check if the sample was answered correctly
            # For acc_norm tasks (cloze), check acc_norm field
            # For acc tasks (MCQA), check acc field
            is_correct = False
            if "acc_norm" in sample:
                is_correct = sample["acc_norm"] == 1.0
            elif "acc" in sample:
                is_correct = sample["acc"] == 1.0

            # Extract relevant information
            sample_data = {
                "task": task_name,
                "doc_id": sample["doc_id"],
                "doc_hash": sample.get("doc_hash", ""),
                "question": sample["doc"]["question"],
                "choices": sample["doc"]["choices"],
                "answer": sample["doc"]["answer"],
                "target": sample["target"],
                "correct": is_correct,
                **const_sample_data,
            }

            # Add any additional fields from doc
            # if "prompt" in sample["doc"]:
            #     sample_data["prompt"] = sample["doc"]["prompt"]
            # if "reasoning" in sample["doc"]:
            #     sample_data["reasoning"] = sample["doc"]["reasoning"]


            samples_data.append(sample_data)

    return samples_data


def process_eval_results(results_dir: Path):
    """Process all evaluation result files in the directory."""
    # Find all pt files
    pt_files = list(results_dir.glob("**/*.pt"))
    print(f"Found {len(pt_files)} pt files")

    if not pt_files:
        return []
    
    # Process each file
    data = []
    for pt_file in tqdm(sorted(pt_files)):
        # Extract global step from filename
        global_step = pt_file.stem

        results = torch.load(pt_file, weights_only=False)
        const_sample_data = {
            "global_step": global_step,
        }
        sample_data = extract_correct_samples(results, const_sample_data)

        data.extend(sample_data)

    return data

def dicts_to_dict(dicts: list[dict], key_field: str) -> dict[str, dict]:
    return {item[key_field]: item for item in dicts}


def main():
    parser = argparse.ArgumentParser(
        description="Extract correctly answered questions from LM eval results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/mnt/ssd-1/lucia/deep-ignorance/analysis/results/evaluations",
        help="Directory containing evaluation results",
    )

    unlearning_checkpoints_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/checkpoints/unlearning_annealing_checkpoints.json")
    with open(unlearning_checkpoints_file, "r") as f:
        full_unlearning_checkpoints = json.load(f)
    
    pretraining_checkpoints = [checkpoint for checkpoint in full_unlearning_checkpoints if checkpoint["stage"] == "pretraining"]
    pretraining_checkpoints = dicts_to_dict(pretraining_checkpoints, "revision")
    annealing_unlearning = [checkpoint for checkpoint in full_unlearning_checkpoints if checkpoint["stage"] == "annealing"]
    annealing_unlearning = dicts_to_dict(full_unlearning_checkpoints, "revision") 

    annealing_checkpoints_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/checkpoints/annealing_unlearning_checkpoints.json")
    with open(annealing_checkpoints_file, "r") as f:
        annealing_unfiltered_checkpoints = json.load(f)
    annealing_unfiltered_checkpoints = dicts_to_dict(annealing_unfiltered_checkpoints, "revision")

    tampering_gradient_difference_checkpoints_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/checkpoints/gradient_difference_tampering_checkpoints.json")
    with open(tampering_gradient_difference_checkpoints_file, "r") as f:
        tampering_gradient_difference_checkpoints = json.load(f)
    tampering_gradient_difference_checkpoints = dicts_to_dict(tampering_gradient_difference_checkpoints, "revision")

    tampering_gradient_ascent_checkpoints_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/checkpoints/gradient_ascent_tampering_checkpoints.json")
    with open(tampering_gradient_ascent_checkpoints_file, "r") as f:
        tampering_gradient_ascent_checkpoints = json.load(f)
    tampering_gradient_ascent_checkpoints = dicts_to_dict(tampering_gradient_ascent_checkpoints, "revision")

    tampering_unfiltered_checkpoints_file = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/checkpoints/unfiltered_tampering_checkpoints.json")
    with open(tampering_unfiltered_checkpoints_file, "r") as f:
        tampering_unfiltered_checkpoints = json.load(f)
    tampering_unfiltered_checkpoints = dicts_to_dict(tampering_unfiltered_checkpoints, "revision")

    model_data = {
        "deep-ignorance-pretraining-stage-unfiltered": {
            "nickname": "pretraining",
            "stage": "pretraining",
            "checkpoints": pretraining_checkpoints,
        },
        "deep-ignorance-unfiltered": {
            "nickname": "annealing",
            "stage": "annealing",
            "checkpoints": annealing_unfiltered_checkpoints,
        },
        "annealing_baseline_ga_v3_interleaved_1_in_50_ga_lr_scale-0.001_gd_lr-0.00012_gclip-0.5": {
            "nickname": "unlearning_annealing",
            "stage": "annealing",
            "checkpoints": annealing_unlearning,
        },
        "annealing_filtered_gdiff_v1_interleav___gclip-0.5-fp-adversarial-20251110_154702": {
            "nickname": "gradient_difference_tampering",
            "stage": "annealing",
            "checkpoints": tampering_gradient_difference_checkpoints,
        },
        "annealing_baseline_ga_v3_interleaved____gclip-0.5-fp-adversarial-20251110_154724": {
            "nickname": "gradient_ascent_tampering",
            "stage": "annealing",
            "checkpoints": tampering_gradient_ascent_checkpoints,
        },
        "deep-ignorance-unfiltered-fp-adversarial-20251110_154700": {
            "nickname": "unfiltered_tampering",
            "stage": "annealing",
            "checkpoints": tampering_unfiltered_checkpoints,
        },
    }
    
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    data = []
    for model_name, model_data in model_data.items():
        const_model_data = {
            "model_name": model_name,
            "nickname": model_data["nickname"],
            "stage": model_data["stage"],
        }
        
        model_results_dir = results_dir / model_name
        raw_data = process_eval_results(model_results_dir)
        checkpoints = model_data["checkpoints"]

        for sample in raw_data:
            data.append({
                **sample, 
                **const_model_data,
                **{"all_stages_step": checkpoints[sample["global_step"]]["all_stages_step"]}
            })

    # Save a combined file with all correct answers
    results_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = results_dir / "all_answers.jsonl"
    with open(results_jsonl, "w") as f:
        for sample in sorted(data, key=lambda x: x["all_stages_step"]):
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved all correct answers to {results_jsonl}")


if __name__ == "__main__":
    main()
