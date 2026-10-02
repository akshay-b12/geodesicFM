"""Rebuild Task 3 summaries and plots from committed image matrices only."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np

from .metric import metric_diagnostics, neighbor_variation
from .estimate_metric import (atomic_json, atomic_npz, fingerprint,
                              read_shard, shard_path)


def write_csv(path, rows):
    if not rows:
        return
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as destination:
            destination.write(handle.getvalue())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def analyze_run(directory, plots=None):
    directory = Path(directory)
    run = json.loads((directory/"run.json").read_text(encoding="utf-8"))
    plan, identity = run["plan"], run["fingerprint"]
    if fingerprint(plan) != identity:
        raise ValueError("Run metadata fingerprint is invalid.")
    if plan["stage"] == "controls":
        raise ValueError("Controls use controls.json, not image aggregation.")
    make_plots = plan["settings"]["plots"] if plots is None else plots
    summary = {"schema_version": 1, "fingerprint": identity, "stage": plan["stage"],
               "status": "COMPLETE", "operators": {}, "split_comparisons": {},
               "interpretation": [
                   "Metrics are unregularized means of realization-level RGB-core Gram matrices.",
                   "Ranks are threshold-dependent numerical diagnostics, not intrinsic dimensions.",
                   "Scaled eigenvectors use xi = S z; raw metric entries retain physical coordinate units.",
                   "Neighbor changes are descriptive fixed-chart differences, not curvature or significance tests.",
                   "Image spread is descriptive; image-level bootstrap and convergence studies remain separate tasks.",
                   "Validation estimates are kept separate and are not pooled into the training metric."]}
    means = {}
    for name, grid in plan["grids"].items():
        summary["operators"][name] = {}
        for split, selection in plan["selections"].items():
            shape = (len(grid["points"]), len(grid["parameter_names"]), len(grid["parameter_names"]))
            image_values, image_ids, crop_counts, noise_spreads = [], [], [], []
            for selected in selection["selected"]:
                image_id = selected["image"]["image_id"]
                path = shard_path(directory, name, split, image_id)
                shard = read_shard(path, identity, image_id, shape)
                if shard is None:
                    raise ValueError(f"Incomplete run: missing committed image {image_id}/{name}. Resume estimation.")
                expected = [r["crop_index"] for r in selected["records"]]
                if shard["crop_indices"].tolist() != expected:
                    raise ValueError("Checkpoint crop IDs differ from the run plan.")
                image_values.append(shard["image_metric"])
                image_ids.append(image_id)
                crop_counts.append(len(expected))
                noise_spreads.append(shard["realization_trace_std"].mean(0))
            values = np.stack(image_values)
            mean = values.mean(0)  # Every image has equal weight.
            means[name, split] = mean
            std = values.std(0, ddof=1) if len(values) > 1 else None
            image_ranks = np.asarray([
                [metric_diagnostics(g, grid["scales"], **plan["settings"]["diagnostics"])["scaled"]["rank"]
                 for g in image] for image in values], dtype=np.int64)
            destination = directory/"analysis"/name/split
            destination.mkdir(parents=True, exist_ok=True)
            atomic_npz(destination/"metric.npz", points=np.asarray(grid["points"]),
                       scales=np.asarray(grid["scales"]), mean_metric=mean,
                       per_image_metric=values, image_ids=np.asarray(image_ids),
                       crop_counts=np.asarray(crop_counts),
                       per_image_scaled_rank=image_ranks,
                       mean_realization_trace_std=np.stack(noise_spreads).mean(0))
            diagnostics, rows = [], []
            for index, g in enumerate(mean):
                d = metric_diagnostics(g, grid["scales"], **plan["settings"]["diagnostics"])
                diagnostics.append({"point_index": index, "point": grid["points"][index], **d})
                row = {"point_index": index, **dict(zip(grid["parameter_names"], grid["points"][index])),
                       "images": len(values), "crops": sum(crop_counts),
                       "image_scaled_rank_min": int(image_ranks[:, index].min()),
                       "image_scaled_rank_max": int(image_ranks[:, index].max()),
                       "image_scaled_full_rank_fraction": float((image_ranks[:, index] == shape[1]).mean())}
                for i, ni in enumerate(grid["parameter_names"]):
                    for j in range(i, shape[1]):
                        nj = grid["parameter_names"][j]
                        row[f"G_{ni}__{nj}"] = float(g[i, j])
                        row[f"image_std_G_{ni}__{nj}"] = float(std[index, i, j]) if std is not None else None
                    row[f"raw_direction_rms_{ni}"] = d["raw"]["direction_rms"][i]
                    row[f"scaled_weak_vector_{ni}"] = d["scaled"]["weak_direction"][i]
                for coordinate in ("raw", "scaled"):
                    result = d[coordinate]
                    row[f"{coordinate}_rank"] = result["rank"]
                    row[f"{coordinate}_rank_threshold"] = result["rank_threshold"]
                    row[f"{coordinate}_condition"] = result["condition_number"]
                    for i, eigenvalue in enumerate(result["eigenvalues"]):
                        row[f"{coordinate}_eigenvalue_{i}"] = eigenvalue
                rows.append(row)
            edges = neighbor_variation(grid["points"], mean, grid["scales"])
            atomic_json(destination/"diagnostics.json", {"grid": grid, "points": diagnostics})
            write_csv(destination/"point_summary.csv", rows)
            # JSON always represents empty edges; CSV exists only for nonempty edges.
            atomic_json(destination/"neighbor_variation.json", edges)
            write_csv(destination/"neighbor_variation.csv", edges)
            ranks = [r["scaled"]["rank"] for r in diagnostics]
            conditions = [r["scaled"]["condition_number"] for r in diagnostics
                          if r["scaled"]["condition_number"] is not None]
            norms = np.linalg.norm(mean*np.asarray(grid["scales"])[None, :, None]
                                   *np.asarray(grid["scales"])[None, None, :], axis=(1, 2))
            summary["operators"][name][split] = {
                "images": len(values), "crops": sum(crop_counts), "points": len(mean),
                "scaled_rank_min": min(ranks), "scaled_rank_max": max(ranks),
                "scaled_singular_points": sum(r < shape[1] for r in ranks),
                "scaled_condition_max_finite": max(conditions) if conditions else None,
                "scaled_metric_norm_min": float(norms.min()), "scaled_metric_norm_max": float(norms.max()),
                "max_neighbor_relative_change": max((e["relative_change"] for e in edges), default=None),
                "analysis_path": str(destination.relative_to(directory))}
            if make_plots:
                from .metric_plots import plot_metric
                plot_metric(destination, grid, mean, diagnostics, split)
            print(f"Analysed {name}/{split}: {len(values)} images, {len(mean)} points, scaled rank {min(ranks)}..{max(ranks)}", flush=True)
        if (name, "train") in means and (name, "valid") in means:
            train, valid = means[name, "train"], means[name, "valid"]
            s = np.asarray(grid["scales"])
            a, b = train*s[None, :, None]*s[None, None, :], valid*s[None, :, None]*s[None, None, :]
            delta = np.linalg.norm(a-b, axis=(1, 2))
            denominator = np.maximum(np.linalg.norm(a, axis=(1, 2)), np.linalg.norm(b, axis=(1, 2)))
            relative = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator > 0)
            summary["split_comparisons"][name] = {
                "scaled_relative_frobenius_difference": relative.tolist(),
                "maximum": float(relative.max()),
                "note": "Descriptive split comparison, without uncertainty estimates or a significance claim."}
    atomic_json(directory/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    analyze_run(args.run, plots=False if args.no_plots else None)


if __name__ == "__main__":
    main()
