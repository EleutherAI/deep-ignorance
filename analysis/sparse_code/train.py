import subprocess

cmds = {
    "0": [
        "torchrun",
        "--master_port",
        "29512",
        "--nproc_per_node",
        "gpu",
        "-m",
        "sparsify",
        "EleutherAI/deep-ignorance-unfiltered",
        "EleutherAI/deep_ignorance_filtered_documents",
        "--data_args",
        "data_dir=pretraining",
        "--batch_size",
        "1",
        "--grad_acc_steps",
        "64",
        "--hookpoints",
        "layers.0.mlp",
        "layers.5.mlp",
        "layers.10.mlp",
        "layers.15.mlp",
        "layers.20.mlp",
        "layers.25.mlp",
        "layers.30.mlp",
        "layers.31.mlp",
        "--optimizer",
        "adam",
        "--transcode",
        "--skip_connection",
        "--expansion_factor",
        "32",
        "--run_name",
        "Deep_Ignorance_Unfiltered_Transcode_Adam_1e-3_k64_e32",
        "--log_to_wandb",
        "True",
        "--k",
        "64",
        "--lr",
        "1e-3",
    ],
    "1": [
        "torchrun",
        "--master_port",
        "29513",
        "--nproc_per_node",
        "gpu",
        "-m",
        "sparsify",
        "EleutherAI/deep-ignorance-e2e-strong-filter",
        "EleutherAI/deep_ignorance_filtered_documents",
        "--data_args",
        "data_dir=pretraining",
        "--batch_size",
        "1",
        "--grad_acc_steps",
        "64",
        "--hookpoints",
        "layers.0.mlp",
        "layers.5.mlp",
        "layers.10.mlp",
        "layers.15.mlp",
        "layers.20.mlp",
        "layers.25.mlp",
        "layers.30.mlp",
        "layers.31.mlp",
        "--optimizer",
        "adam",
        "--transcode",
        "--skip_connection",
        "--expansion_factor",
        "32",
        "--run_name",
        "Deep_Ignorance_Filtered_Transcode_Adam_1e-3_k64_e32",
        "--log_to_wandb",
        "True",
        "--k",
        "64",
        "--lr",
        "1e-3",
    ]
}


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--id", type=str, required=True, default="0")
    args = parser.parse_args()

    print(" ".join(cmds[args.id]))

    result = subprocess.run(
        cmds[args.id],
        text=True,
        capture_output=True,
    )

    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()