import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from analysis.utils import assert_type


def filter_unstably_learned_questions(pretrain: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove items that are not correct at the final pretraining step."""
    pretrain = pretrain.sort_values('all_stages_step')

    # Identify final-step correctness per (task, doc_id) pair
    final_pretrain_correct = pretrain.groupby(["task", "doc_id"])["correct"].last()
    stable_pretrain_ids = final_pretrain_correct[final_pretrain_correct].index

    # Build stable-only dataframe
    stable_pretrain = (
        pretrain.set_index(["task", "doc_id"])
               .loc[stable_pretrain_ids]
               .reset_index()
    )
    stable_pretrain = assert_type(pd.DataFrame, stable_pretrain)

    if verbose:
        final_step = pretrain['all_stages_step'].max()
        final_step_data = pretrain[pretrain['all_stages_step'] == final_step]
        print(f"Final pretraining step: {final_step}")
        print(f"Correct at final pretraining step: {final_step_data['correct'].sum()} / {len(final_step_data)}")

        # Use the same unit for counts: unique (task, doc_id) pairs
        total_pairs = len(final_pretrain_correct)  # same as pretrain.groupby(["task","doc_id"]).ngroups
        unstable_pretrain_ids = set(final_pretrain_correct.index) - set(stable_pretrain_ids)

        ever_correct = pretrain.groupby(["task", "doc_id"])["correct"].any()
        transient_learners = ever_correct[ever_correct].index.difference(stable_pretrain_ids)

        print(
            f"Excluded {len(unstable_pretrain_ids)} unstably learned pretraining questions "
            f"(out of {total_pairs}); "
            f"{len(transient_learners)} of these correct at step(/s) other than the final one."
        )

    return stable_pretrain


def filter_unstably_unlearned_questions(unlearn: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove items that are not incorrect at the final unlearning step."""
    unlearn = unlearn.sort_values('all_stages_step')

    final_unlearn_incorrect = (
        ~unlearn.groupby(["task", "doc_id"])["correct"].last()
    )
    stable_unlearn_ids = final_unlearn_incorrect[final_unlearn_incorrect].index

    stable_unlearn = (
        unlearn.set_index(["task", "doc_id"])
               .loc[stable_unlearn_ids]
               .reset_index()
    )
    stable_unlearn = assert_type(pd.DataFrame, stable_unlearn)

    if verbose:
        unstable_unlearn_ids = set(final_unlearn_incorrect.index) - set(stable_unlearn_ids)
        print(f"Excluded {len(unstable_unlearn_ids)} unstably unlearned questions "
        f"(out of {len(final_unlearn_incorrect)}).")

    return stable_unlearn


def plot_correct_over_time(df: pd.DataFrame):
    """Plot number of correct questions over training steps, grouped by nickname."""
    plt.figure(figsize=(10, 6))

    # Ensure consistent ordering of phases
    phase_order = ["pretraining", "annealing", "unlearning_annealing"]

    for name in sorted(df["nickname"].unique(), key=lambda x: phase_order.index(x) if x in phase_order else 999):
        group = df[df["nickname"] == name]
        counts = (
            group.groupby("all_stages_step")["correct"]
            .sum()
            .reset_index()
            .sort_values("all_stages_step")
        )
        if counts.empty:
            continue
        plt.plot(
            counts["all_stages_step"],
            counts["correct"],
            label=name.replace("_", " "),
            linewidth=2,
            marker="o",
            markersize=3,
            alpha=0.8,
        )

    plt.xlabel("Training Step (all_stages_step)")
    plt.ylabel("Number of Questions Correct")
    plt.title("Number of Questions Correct Over Training Steps")
    plt.legend(title="Phase")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save before showing
    plt.savefig("correct_over_time.png", dpi=300)
    plt.show()





# def plot_correct_over_time(df: pd.DataFrame):
#     """Plot number of correct questions over training steps, grouped by nickname."""
#     plt.figure(figsize=(10, 6))

#     for name, group in df.groupby("nickname"):
#         counts = (
#             group.groupby("all_stages_step")["correct"]
#             .sum()
#             .reset_index()
#             .sort_values("all_stages_step")
#         )
#         plt.plot(
#             counts["all_stages_step"],
#             counts["correct"],
#             label=name.replace("_", " "),
#             linewidth=2,
#         )

#     plt.xlabel("Training Step (all_stages_step)")
#     plt.ylabel("Number of Questions Correct")
#     plt.title("Number of Questions Correct Over Training Steps")
#     plt.legend(title="Phase")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     plt.savefig("correct_over_time.png")


def main():
    df = pd.read_json(
        "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/evaluations/all_answers.jsonl",
        lines=True,
    )
    plot_correct_over_time(df)

    test = True

    if test:
        # For pretraining: find questions that transition from incorrect to correct
        pretrain = df[df['stage'] == 'pretraining'].sort_values('all_stages_step')
        pretrain = filter_unstably_learned_questions(pretrain)
        learning = pretrain.groupby('doc_id', group_keys=False).apply(
            lambda x: x[x['correct']]['all_stages_step'].min() if (~x['correct']).any() and x['correct'].any() else None, 
            include_groups=False
        ).dropna()

        # For unlearning: find questions that transition from correct to incorrect
        unlearn = df[df['nickname'] == 'unlearning_annealing'].sort_values('all_stages_step')
        unlearn = filter_unstably_unlearned_questions(unlearn)
        unlearning = unlearn.groupby('doc_id', group_keys=False).apply(
            lambda x: x[~x['correct']]['all_stages_step'].min() if x['correct'].any() and (~x['correct']).any() else None,
            include_groups=False
        ).dropna()

        # Find common questions that both learned and unlearned
        common_docs = learning.index.intersection(unlearning.index)
        learning_steps = learning[common_docs].sort_index()
        unlearning_steps = unlearning[common_docs].sort_index()

        learning_order = learning_steps.rank()
        unlearning_order = unlearning_steps.rank()

        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(learning_order, unlearning_order)
        pearson_corr, pearson_p = pearsonr(learning_order, unlearning_order)

        print(f"Number of questions analyzed: {len(common_docs)}")
        print(f"\nSpearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
        print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
        print(f"\nInterpretation: {'Significant' if spearman_p < 0.05 else 'Not significant'} relationship between learning and unlearning order")

    


if __name__ == "__main__":
    main()