"""Compact scientific report from Task 4 diagnostics."""
from pathlib import Path


def write_report(path, summary):
    lines = ["# Task 4: metric variation and eigenstructure", "",
             f"Source stage: **{summary['source_stage']}**. Status: **{summary['status']}**.", "",
             "This is a descriptive analysis of the supplied Task 3 matrices. It does not establish statistical convergence, intrinsic dimension, curvature or global identifiability.", "",
             "Training and validation remain separate. Matrices are preserved without jitter, clipping or inversion. Ranks use saved explicit thresholds in fixed raw/scaled coordinates.", "",
             "| Operator | Split | Images | Points | Scaled local rank | Global-reference rank | Threshold-sensitive points | Max neighbor relative change |",
             "|---|---|---:|---:|---|---|---:|---:|"]
    for operator, splits in summary["operators"].items():
        for split, item in splits.items():
            change = item["max_neighbor_relative_change"]
            change_text = f"{change:.4g}" if change is not None else "unavailable"
            lines.append(f"| {operator} | {split} | {item['images']} | {item['points']} | {item['local_rank_min']}–{item['local_rank_max']} | {item['global_rank_min']}–{item['global_rank_max']} | {item['rank_sweep_sensitive_points']} | {change_text} |")
    lines += ["", "## Interpretation", "",
              "- Diagonal entries describe squared RMS coordinate sensitivities; off-diagonal entries describe derivative alignment. Undefined correlations are null, including zero directions.",
              "- Raw entries use physical parameters. Scaled analysis uses xi = S z with the original fixed characteristic scales; scaled G is S G S. Eigenvalues and conditions depend on this coordinate choice.",
              "- Local rank uses each point's largest eigenvalue. Global-reference rank uses the largest eigenvalue across the analyzed operator/split grid. Neither is an intrinsic-dimension estimate.",
              "- Threshold sensitivity identifies dependence on numerical reporting choices. Agreement across the sweep is not a confidence interval or proof of exact rank.",
              "- The weakest eigenspace is clustered using the declared eigengap tolerance. An individual weak vector is reported only when that space is one-dimensional. Angle comparisons are sign/basis invariant and are unavailable when compared space dimensions differ.",
              "- One-dimensional full-rank matrices have condition one regardless of absolute sensitivity. Read eigenvalue magnitude and RMS sensitivity as well.",
              "- Image deviations and image eigenstructure agreement are descriptive. Averaging can increase rank because images can reveal different directions. Bootstrap/content convergence remains Task 5.",
              "- Neighbor changes use adjacent sampled axial points only. Relative change divides by the larger endpoint scaled Frobenius norm. Variation is not curvature. No unsampled boundary is inferred.",
              "- Eigenvalue fractions and participation ratio use nonnegative display contributions only. Stored eigenvalues, including tolerated roundoff-negative ones, remain unchanged. Participation ratio is not intrinsic dimension.", ""]
    if summary["split_comparisons"]:
        lines += ["## Split comparisons", ""]
        for operator, item in summary["split_comparisons"].items():
            lines.append(f"- {operator}: maximum scaled relative train/validation difference = {item['max_relative_difference']:.4g}; comparable weakest-space points = {item['comparable_weak_space_points']}/{item['points']}. These differences have no significance interpretation here.")
    controls = summary.get("controls")
    if controls:
        lines += ["", "## Synthetic controls", "",
                  f"Imported {controls['cases']} passing Task 3 control cases in {controls['groups']} operator/pattern groups. These are mathematical sanity checks and are not DIV2K measurements."]
    else:
        lines += ["", "Synthetic controls were not supplied for this analysis run."]
    lines += ["", "## What can be concluded now", "",
              "The output identifies observed local sensitivities, numerical weak directions, spectral degeneracies and changes at the supplied points. A pilot supports implementation checks and exploratory observations. Final geometry decisions require full-data stability and the subsequent invariance/curvature/geodesic checks.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
