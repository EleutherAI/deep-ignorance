import socket
import json
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes

import torch
from transformers import AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM


@torch.inference_mode()
def evaluate(
    rank: int,
    model_name: str,
    model_revision: str,
    lm_eval_tasks_path: Path,
    results_file: Path,
):
    """Non-Claude re-implementation of an entire directory in one file."""
    results_file.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision)
    model = model.to(device)

    model.eval()
    model = HFLM(model)

    results = evaluator.simple_evaluate(
        model=model,
        model_args=f"revision={model_revision}",
        tasks=["wmdp_bio_cloze_verified", "wmdp_bio_robust"],
        log_samples=True,
        task_manager=TaskManager(include_path=str(lm_eval_tasks_path)),
        device=device,
    )

    torch.save(results, results_file)


def distribute_evaluations_single_node(
    model_name: str, revisions: list[str], results_dir: Path
):
    lm_eval_tasks_path = Path("/mnt/ssd-1/lucia/deep-ignorance/lm_eval_tasks")

    world_size = torch.cuda.device_count()
    print(f"Number of GPUs: {world_size}")
    
    for i in range(0, len(revisions), world_size):
        batch = revisions[i : i + world_size]

        if all([
            (results_dir / model_name.split('/')[-1] / f"{revision}.pt").exists()
            for revision in batch
        ]):
            print(f"Skipping already evaluated revisions: {batch}")
            continue

        if world_size <= 1:
            # Run the worker directly if no distributed training is needed. This is great
            # for debugging purposes.
            evaluate(0, model_name, revisions[i], lm_eval_tasks_path, results_pt_file)
        else:
            args = {}
            for rank, revision in enumerate(batch):
                results_pt_file = results_dir / model_name.split('/')[-1] / f"{revision}.pt"
                # if results_pt_file.exists():
                    # print(f"Skipping already evaluated revision: {revision}")
                    # continue

                args[rank] = (rank, model_name, revision, lm_eval_tasks_path, results_pt_file)

            # Set up multiprocessing and distributed training
            mp.set_sharing_strategy("file_system")

            # Find an available port for distributed training
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                _, port = s.getsockname()

            ctx = start_processes(
                "evaluate",
                evaluate,
                args=args,
                envs={
                    rank: {
                        "LOCAL_RANK": str(rank),
                        "MASTER_ADDR": "localhost",
                        "MASTER_PORT": str(port),
                    }
                    for rank in args.keys()
                },
                logs_specs=DefaultLogsSpecs(),
            )
            ctx.wait()


def main():
    # Source
    checkpoints_file = "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/available_checkpoints.json"
    with open(checkpoints_file, "r") as f:
        checkpoints = json.load(f)

    # Destination
    results_dir = Path("/mnt/ssd-1/lucia/deep-ignorance/analysis/results/evaluations")

    # Models to analyze
    model_names = [
        "EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        "EleutherAI/deep-ignorance-unfiltered",
        "EleutherAI/annealing_baseline_ga_v3_interleaved_1_in_50_ga_lr_scale-0.001_gd_lr-0.00012_gclip-0.5",
        "EleutherAI/deep-ignorance-unfiltered-fp-adversarial-20251110_154700",
        "EleutherAI/annealing_filtered_gdiff_v1_interleav___gclip-0.5-fp-adversarial-20251110_154702",
        "EleutherAI/annealing_baseline_ga_v3_interleaved____gclip-0.5-fp-adversarial-20251110_154724",
    ]

    for model_name in model_names:
        revisions = [
            checkpoint['revision'] for checkpoint in checkpoints[model_name]
        ]
        print(f"Evaluating {model_name} with {len(revisions)} revisions")
        print(f"Revisions: {revisions}")

        distribute_evaluations_single_node(
            model_name=model_name,
            revisions=revisions,
            results_dir=results_dir
        )
    

if __name__ == "__main__":
    main()
    