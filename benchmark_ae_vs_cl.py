"""
Benchmark: Autoencoder vs Chow-Liu Tree for outlier detection.

Computes ROC AUC, precision@k, and lift for both methods using the
composite outlier indicator (respondents with 3+ simultaneous extreme
food/body-measurement values) as ground truth.

Runs on both SADC 2015 and 2017 datasets with multiple AE configurations.

Usage:
    python benchmark_ae_vs_cl.py
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset.loader import DataLoader
from utils import define_necessary_elements


def load_ground_truth(dataset_name):
    """Load dataset and build composite outlier ground truth labels."""
    (
        drop_columns, rename_columns, interest_columns,
        additional_drop_columns, additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(dataset_name, None, None, None)

    data_loader = DataLoader(
        drop_columns, rename_columns, interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    outlier_labels = data_loader.find_outlier_data_sadc_2017(dataset_name, ["outlier"])
    return outlier_labels


def compute_metrics(error_scores, ground_truth, method_name):
    """Compute evaluation metrics for a given method."""
    common_idx = error_scores.index.intersection(ground_truth.index)
    errors = error_scores.loc[common_idx]
    labels = ground_truth.loc[common_idx, "outlier"].astype(int)

    n_total = len(labels)
    n_outliers = int(labels.sum())
    prevalence = n_outliers / n_total

    # Normalize error scores to [0, 1]
    min_e, max_e = errors["error"].min(), errors["error"].max()
    if max_e > min_e:
        scores = (errors["error"] - min_e) / (max_e - min_e)
    else:
        scores = errors["error"]

    # ROC AUC and AP require both classes present
    n_classes = labels.nunique()
    if n_classes < 2:
        auc = None
        ap = None
    else:
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

    # Sort by error descending (most anomalous first)
    sorted_idx = scores.sort_values(ascending=False).index
    sorted_labels = labels.loc[sorted_idx].values

    results = {
        "method": method_name,
        "n_total": n_total,
        "n_outliers": n_outliers,
        "prevalence": round(prevalence, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
        "avg_precision": round(ap, 4) if ap is not None else "N/A",
    }

    for k in [10, 25, 50, 100, 200, n_outliers]:
        if k <= n_total:
            precision_k = sorted_labels[:k].sum() / k
            lift_k = precision_k / prevalence if prevalence > 0 else 0
            label = k if k != n_outliers else f"{k}(=n_outliers)"
            results[f"precision@{label}"] = round(float(precision_k), 4)
            results[f"lift@{label}"] = round(float(lift_k), 2)

    for k in [100, 200, 500]:
        if k <= n_total:
            recall_k = sorted_labels[:k].sum() / n_outliers if n_outliers > 0 else 0
            results[f"recall@{k}"] = round(float(recall_k), 4)

    return results


def print_comparison(dataset_name, all_results):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {dataset_name}")
    print(f"{'=' * 80}")
    print("Ground truth: composite indicator (3+ simultaneous extreme values)")

    if not all_results:
        print("  No results to display.")
        return

    # Collect all keys
    all_keys = []
    for r in all_results:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    # Print as table
    methods = [r["method"] for r in all_results]
    col_width = max(20, max(len(m) for m in methods) + 2)
    key_width = max(len(k) for k in all_keys)

    header = f"{'Metric':<{key_width}}"
    for m in methods:
        header += f"  {m:>{col_width}}"
    print(f"\n{header}")
    print("-" * len(header))

    for k in all_keys:
        if k == "method":
            continue
        row = f"{k:<{key_width}}"
        for r in all_results:
            val = r.get(k, "N/A")
            row += f"  {str(val):>{col_width}}"
        print(row)


def run_benchmark_for_dataset(dataset_name, error_files):
    """Run benchmark for a single dataset with multiple methods."""
    print(f"\nLoading ground truth for {dataset_name}...")
    gt = load_ground_truth(dataset_name)
    print(f"  Total rows: {len(gt)}, Outliers (label=1): {int(gt['outlier'].sum())}")

    all_results = []
    for method_name, error_path in error_files:
        if not os.path.exists(error_path):
            print(f"  SKIP {method_name}: {error_path} not found")
            continue
        errors = pd.read_csv(error_path)
        print(f"  {method_name}: {len(errors)} rows loaded from {error_path}")
        metrics = compute_metrics(errors, gt, method_name)
        all_results.append(metrics)

    print_comparison(dataset_name, all_results)
    return all_results


def main():
    all_results = []

    # ── SADC 2017 ──
    sadc2017_methods = [
        ("AE (latent=8, 50ep)", "cache/benchmark/ae_sadc2017_outliers/errors.csv"),
        ("AE (latent=2, 5ep)", "cache/benchmark/ae_sadc2017_default_outliers/errors.csv"),
        ("Chow-Liu (α=1.0)", "cache/benchmark/cl_sadc2017/errors.csv"),
    ]
    results_2017 = run_benchmark_for_dataset("sadc_2017", sadc2017_methods)
    for r in results_2017:
        r["dataset"] = "sadc_2017"
    all_results.extend(results_2017)

    # ── SADC 2015 ──
    sadc2015_methods = [
        ("AE (latent=8, 50ep)", "cache/benchmark/ae_sadc2015_outliers/errors.csv"),
        ("Chow-Liu (α=1.0)", "cache/benchmark/cl_sadc2015/errors.csv"),
    ]
    results_2015 = run_benchmark_for_dataset("sadc_2015", sadc2015_methods)
    for r in results_2015:
        r["dataset"] = "sadc_2015"
    all_results.extend(results_2015)

    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        out_path = "cache/benchmark/benchmark_results_all.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nAll results saved to {out_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for r in all_results:
        print(f"  {r.get('dataset','?'):12s}  {r['method']:30s}  AUC={r['roc_auc']}  AP={r['avg_precision']}")


if __name__ == "__main__":
    main()
