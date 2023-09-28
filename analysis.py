import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from evaluator import Evaluator


def compute_stats(group):
    precision, recall, f1, _ = precision_recall_fscore_support(
        group["label"], group["pred"], average="binary"
    )
    accuracy = (group["label"] == group["pred"]).mean()
    return pd.Series(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": len(group),
        }
    )


def process_results(
    data: dict, results: list[dict], evaluator: Evaluator
) -> dict[str, pd.DataFrame]:
    # Process results
    baseline_results = []
    experiment_results = []
    for _, res, metadata in results:
        if metadata["trial_type"] == "baseline":
            baseline_results.append(
                {
                    "rule_id": metadata["rule_id"],
                    "rule": data["rules"][metadata["rule_id"]]["rule"],
                    "example": metadata["example"],
                    "label": metadata["label"],
                    "pred": int(
                        evaluator.check_rejection(
                            res["choices"][0]["message"]["content"]
                        )
                    ),
                }
            )
        elif metadata["trial_type"] == "experiment":
            experiment_results.append(
                {
                    "num_rules": metadata["num_rules"],
                    "trial": metadata["trial"],
                    "label": metadata["label"],
                    "pred": int(
                        evaluator.check_rejection(
                            res["choices"][0]["message"]["content"]
                        )
                    ),
                }
            )

    baseline_results_df = pd.DataFrame(baseline_results)
    baseline_stats_df = baseline_results_df.groupby(["rule_id", "rule"]).apply(
        compute_stats
    )
    baseline_stats_df = baseline_stats_df.sort_values(by="f1", ascending=False)

    experiment_results_df = pd.DataFrame(experiment_results)
    experiment_stats_df = experiment_results_df.groupby(["num_rules", "trial"]).apply(
        compute_stats
    )
    experiment_stats_df = experiment_stats_df.sort_values(by=["num_rules", "trial"])

    return {
        "baseline_results": baseline_results_df,
        "baseline_stats": baseline_stats_df,
        "experiment_results": experiment_results_df,
        "experiment_stats": experiment_stats_df,
    }


def combine_results(
    results_list: list[dict],
    result_type: str,
    group_by_columns: list[str],
    order: list[str],
):
    results = [
        result[result_type].assign(run_key=result["run_key"]) for result in results_list
    ]
    results_df = pd.concat(results, ignore_index=True)
    stats_df = results_df.groupby(group_by_columns).apply(compute_stats).reset_index()
    stats_df["run_key"] = pd.Categorical(
        stats_df["run_key"], categories=order, ordered=True
    )
    stats_df = stats_df.sort_values("run_key")
    return results_df, stats_df
