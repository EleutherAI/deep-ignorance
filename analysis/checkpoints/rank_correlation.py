import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from analysis.utils import assert_type


# def get_stably_learned_step(question_df: pd.DataFrame):
#     incorrect_steps = question_df[~question_df["correct"]]["all_stages_step"]
#     correct_steps = question_df[question_df["correct"]]["all_stages_step"]
    
#     if len(incorrect_steps) == 0:
#         # Always correct, return first correct step
#         return correct_steps.min()
    
#     last_incorrect = incorrect_steps.max()
#     steps_after_last_incorrect = correct_steps[correct_steps > last_incorrect]
    
#     if len(steps_after_last_incorrect) > 0:
#         return steps_after_last_incorrect.min()

#     # Always incorrect, filter out
#     return None


def plot_correct_over_time(df: pd.DataFrame):
    """Plot number of correct questions over training steps, grouped by nickname."""
    plt.figure(figsize=(10, 6))

    # Ensure consistent ordering of phases
    phase_order = ["pretraining", "annealing", "unlearning_annealing"]

    for name in sorted(
        df["nickname"].unique(),
        key=lambda x: phase_order.index(x) if x in phase_order else 999,
    ):
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


def spearman_rank_correlation(subset_label, subset_docs, learning_steps, unlearning_steps):
    learning_sub = learning_steps.loc[subset_docs]
    unlearning_sub = unlearning_steps.loc[subset_docs]

    assert not learning_sub.index.duplicated().any()
    assert not unlearning_sub.index.duplicated().any()
    assert len(learning_sub) == len(unlearning_sub)

    if len(subset_docs) < 3:
        print(
            f"\n[{subset_label}] Too few items ({len(subset_docs)}) for reliable correlation."
        )
        return

    learning_order = learning_sub.rank()
    unlearning_order = unlearning_sub.rank()
    spearman_corr, spearman_p = spearmanr(learning_order, unlearning_order)
    # pearson_corr, pearson_p = pearsonr(learning_order, unlearning_order)
    print(f"\n[{subset_label}]")
    print(f"  Number of questions: {len(subset_docs)}")
    print(f"  Spearman ρ = {spearman_corr:.4f} (p={spearman_p:.4e})")
    # The same in expectation for comparing ranks. Compare actual values for differences.
    # print(f"  Pearson  r = {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(spearman_p, type(spearman_p))
    if spearman_p < 0.05:
        print(
            f"  → Significant relationship between item orders"
        )
    else:
        print("  → Not significant")


def spearman_by_task(first, second):
    # Intersection of learned and unlearned questions
    common_docs = first.index.intersection(second.index)
    learning_steps = first[common_docs].sort_index()
    unlearning_steps = second[common_docs].sort_index()

    # Split by task type
    cloze_docs = common_docs[
        common_docs.get_level_values("task") == "wmdp_bio_cloze_verified"
    ]
    other_docs = common_docs[
        common_docs.get_level_values("task") != "wmdp_bio_cloze_verified"
    ]

    # Global and split analyses
    print(f"Overall: {len(common_docs)} common questions total")
    spearman_rank_correlation("All tasks (combined)", common_docs, learning_steps, unlearning_steps)
    spearman_rank_correlation("WMDP Bio Cloze Verified", cloze_docs, learning_steps, unlearning_steps)
    spearman_rank_correlation("WMDP Bio Robust MCQA", other_docs, learning_steps, unlearning_steps)


# def filter_unstably_learned_questions(
#     pretrain: pd.DataFrame, verbose: bool = True
# ) -> pd.DataFrame:
#     """Remove items that are not correct at the final pretraining step."""
#     pretrain = pretrain.sort_values("all_stages_step")

#     # Identify final-step correctness per (task, doc_id) pair
#     final_pretrain_correct = pretrain.groupby(["task", "doc_id"])["correct"].last()
#     stable_pretrain_ids = final_pretrain_correct[final_pretrain_correct].index

#     # Build stable-only dataframe
#     stable_pretrain = (
#         pretrain.set_index(["task", "doc_id"]).loc[stable_pretrain_ids].reset_index()
#     )
#     stable_pretrain = assert_type(pd.DataFrame, stable_pretrain)

#     if verbose:
#         final_step = pretrain["all_stages_step"].max()
#         final_step_data = pretrain[pretrain["all_stages_step"] == final_step]
#         print(f"Final pretraining step: {final_step}")
#         print(
#             f"Correct at final pretraining step: {final_step_data['correct'].sum()} / {len(final_step_data)}"
#         )

#         # Use the same unit for counts: unique (task, doc_id) pairs
#         total_pairs = len(
#             final_pretrain_correct
#         )  # same as pretrain.groupby(["task","doc_id"]).ngroups
#         unstable_pretrain_ids = set(final_pretrain_correct.index) - set(
#             stable_pretrain_ids
#         )

#         ever_correct = pretrain.groupby(["task", "doc_id"])["correct"].any()
#         transient_learners = ever_correct[ever_correct].index.difference(
#             stable_pretrain_ids
#         )

#         print(
#             f"Excluded {len(unstable_pretrain_ids)} unstably learned pretraining questions "
#             f"(out of {total_pairs}); "
#             f"{len(transient_learners)} of these correct at step(/s) other than the final one."
#         )

#     return stable_pretrain


# def filter_unstably_unlearned_questions(
#     unlearn: pd.DataFrame, verbose: bool = True
# ) -> pd.DataFrame:
#     """Remove items that are not incorrect at the final unlearning step."""
#     unlearn = unlearn.sort_values("all_stages_step")

#     final_unlearn_incorrect = ~unlearn.groupby(["task", "doc_id"])["correct"].last()
#     stable_unlearn_ids = final_unlearn_incorrect[final_unlearn_incorrect].index

#     stable_unlearn = (
#         unlearn.set_index(["task", "doc_id"]).loc[stable_unlearn_ids].reset_index()
#     )
#     stable_unlearn = assert_type(pd.DataFrame, stable_unlearn)

#     if verbose:
#         unstable_unlearn_ids = set(final_unlearn_incorrect.index) - set(
#             stable_unlearn_ids
#         )
#         print(
#             f"Excluded {len(unstable_unlearn_ids)} unstably unlearned questions "
#             f"(out of {len(final_unlearn_incorrect)})."
#         )

#     return stable_unlearn


def get_stable_learn_step(question_df: pd.DataFrame):
    """Get the first step where correct after the last step where incorrect."""
    incorrect_steps = question_df[~question_df["correct"]]["all_stages_step"]
    correct_steps = question_df[question_df["correct"]]["all_stages_step"]
    
    if len(incorrect_steps) == 0:
        # Always correct, filter out
        return None
    
    last_incorrect = incorrect_steps.max()
    steps_after_last_incorrect = correct_steps[correct_steps > last_incorrect]
    
    if len(steps_after_last_incorrect) > 0:
        return steps_after_last_incorrect.min()

    # Always incorrect, filter out
    return None


def get_stable_unlearn_step(question_df: pd.DataFrame):
    """Get the first step where incorrect after the last step where correct."""
    correct_steps = question_df[question_df["correct"]]["all_stages_step"]
    incorrect_steps = question_df[~question_df["correct"]]["all_stages_step"]
    
    if len(correct_steps) == 0:
        # Always incorrect, filter out
        return None
    
    last_correct = correct_steps.max()
    steps_after_last_correct = incorrect_steps[incorrect_steps > last_correct]
    
    if len(steps_after_last_correct) > 0:
        return steps_after_last_correct.min()

    # Always correct, filter out
    return None


def main():
    df = pd.read_json(
        "/mnt/ssd-1/lucia/deep-ignorance/analysis/results/evaluations/all_answers.jsonl",
        lines=True,
    )
    plot_correct_over_time(df)

    # Pretraining learning steps
    pretrain = df[df["nickname"] == "pretraining"].sort_values("all_stages_step") # type: ignore[arg-type]
    learning = pretrain.groupby(["task", "doc_id"], group_keys=False).apply(
        get_stable_learn_step, include_groups=False
    ).dropna()
    print("learning head:\n", learning.head())

    # Annealing unlearning steps
    print("Spearman correlation between learning and unlearning:")
    unlearn = df[df["nickname"] == "unlearning_annealing"].sort_values(
        "all_stages_step" # type: ignore[arg-type]
    )
    unlearning = unlearn.groupby(["task", "doc_id"], group_keys=False).apply(
        get_stable_unlearn_step, include_groups=False
    ).dropna()
    print("unlearning head:\n", unlearning.head())

    # Intersection of learned and unlearned questions
    common_docs = learning.index.intersection(unlearning.index)
    spearman_by_task(learning[common_docs], unlearning[common_docs])

    # Tampering gradient ascent steps
    print("Spearman correlation between learning and GA tampering:")
    ga_tamper = df[df["nickname"] == "gradient_ascent_tampering"].sort_values(
        "all_stages_step" # type: ignore[arg-type]
    )
    ga_tampering = ga_tamper.groupby(["task", "doc_id"], group_keys=False).apply(
        get_stable_learn_step, include_groups=False
    ).dropna()
    print("ga_tampering head:\n", ga_tampering.head())

    # Intersection of learned and GA tampering steps
    common_docs = learning.index.intersection(ga_tampering.index)
    spearman_by_task(learning[common_docs], ga_tampering[common_docs])

    # Tampering gradient difference steps
    print("Spearman correlation between learning and gradient difference tampering:")
    gd_tamper = df[df["nickname"] == "gradient_difference_tampering"].sort_values(
        "all_stages_step" # type: ignore[arg-type]
    )
    gd_tampering = gd_tamper.groupby(["task", "doc_id"], group_keys=False).apply(
        get_stable_learn_step, include_groups=False
    ).dropna()
    print("gd_tampering head:\n", gd_tampering.head())

    # Intersection of learned and GD tampering steps
    common_docs = learning.index.intersection(gd_tampering.index)
    spearman_by_task(learning[common_docs], gd_tampering[common_docs])


if __name__ == "__main__":
    main()
