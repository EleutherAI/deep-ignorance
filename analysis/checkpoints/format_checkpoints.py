#!/usr/bin/env python3
import json

def format_checkpoints(
    input_path: str,
    output_path: str,
    hf_models: dict[str, str]
):
    """
    Reformats checkpoint sequence.
    """
    with open(input_path, 'r') as f:
        checkpoints_data = json.load(f)

    corrected_checkpoints = []

    # Phase 1: Add ALL pretraining checkpoints first (these come first in time)
    assert "pretraining" in hf_models

    for checkpoint in sorted(checkpoints_data[hf_models["pretraining"]], key=lambda x: x["step"]):
        corrected_checkpoints.append({
            "step": checkpoint["step"],
            "model_name": hf_models["pretraining"],
            "revision": checkpoint["revision"],
            "stage": "pretraining",
            "temporal_order": len(corrected_checkpoints),
            "all_stages_step": checkpoint["step"],
        })

    print(f"Added {len(corrected_checkpoints)} pretraining checkpoints")
    print(f"Pretraining range: {corrected_checkpoints[0]['step']} → {corrected_checkpoints[-1]['step']}")

    # Phase 2: Add ALL annealing checkpoints second (these come after ALL pretraining)
    pretraining_final_step = max(ckpt["step"] for ckpt in checkpoints_data[hf_models["pretraining"]])

    if "annealing" in hf_models:
        annealing_start_idx = len(corrected_checkpoints)
        
        for checkpoint in sorted(checkpoints_data[hf_models["annealing"]], key=lambda x: x["step"]):
            corrected_checkpoints.append({
                "step": checkpoint["step"],
                "model_name": hf_models["annealing"],
                "revision": checkpoint["revision"],
                "stage": "annealing",
                "temporal_order": len(corrected_checkpoints),
                "all_stages_step": checkpoint["step"] + pretraining_final_step,
            })

        annealing_checkpoints = corrected_checkpoints[annealing_start_idx:]
        print(f"Added {len(annealing_checkpoints)} annealing checkpoints")
        print(f"Annealing range: {annealing_checkpoints[0]['step']} → {annealing_checkpoints[-1]['step']}")
    
    # Alternate annealing phase checkpoints
    if "unlearning_annealing" in hf_models:
        unlearning_annealing_start_idx = len(corrected_checkpoints)

        for checkpoint in sorted(checkpoints_data[hf_models["unlearning_annealing"]], key=lambda x: x["step"]):
            corrected_checkpoints.append({
                "step": checkpoint["step"],
                "model_name": hf_models["unlearning_annealing"],
                "revision": checkpoint["revision"],
                "stage": "annealing",
                "temporal_order": len(corrected_checkpoints),
                "all_stages_step": checkpoint["step"] + pretraining_final_step,
            })

        unlearning_annealing_checkpoints = corrected_checkpoints[unlearning_annealing_start_idx:]
        print(f"Added {len(unlearning_annealing_checkpoints)} annealing unlearning checkpoints")
        print(f"Annealing unlearning range: {unlearning_annealing_checkpoints[0]['step']} → {unlearning_annealing_checkpoints[-1]['step']}")

    print(f"\nTotal checkpoints: {len(corrected_checkpoints)}")

    output_file = output_path
    with open(output_file, 'w') as f:
        json.dump(corrected_checkpoints, f, indent=2)
    print(f"\n✅ Saved corrected sequence to {output_file}")