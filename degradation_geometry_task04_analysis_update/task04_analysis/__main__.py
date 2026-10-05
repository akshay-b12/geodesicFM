"""Task 4 CLI: python -m task04_analysis --run outputs/task03/pilot."""
import argparse
from pathlib import Path
import platform
import numpy as np

from .io import (load_settings, load_run, load_matrices, load_controls, fingerprint,
                 sha256, atomic_json, write_csv, read_json, safe_name)
from .numerics import analyze_field, describe_images, symmetric_relative_change, subspace_change
from .report import write_report


def flatten_points(grid, field, images):
    rows = []
    names, scales = grid["parameter_names"], grid["scales"]
    for point, image in zip(field["diagnostics"], images):
        row = {"point_index": point["point_index"], **dict(zip(names, point["point"])),
               "reference_relative_change": point["reference_relative_change"],
               "image_count": image["images"], "image_rank_min": image["scaled_image_rank_min"],
               "image_rank_max": image["scaled_image_rank_max"],
               "image_full_rank_fraction": image["scaled_image_full_rank_fraction"],
               "mean_relative_image_deviation": image["mean_relative_image_deviation"],
               "max_image_weak_space_angle_degrees": image["max_image_weak_space_angle_degrees"]}
        for coordinate in ("raw", "scaled"):
            d = point[coordinate]
            for key in ("rank", "rank_threshold", "global_reference_rank", "global_rank_threshold",
                        "condition_number", "smallest_to_largest_eigenvalue_ratio", "participation_ratio",
                        "weak_subspace_dimension", "max_abs_offdiagonal_correlation", "negative_roundoff_eigenvalues"):
                row[f"{coordinate}_{key}"] = d[key]
            for j, name in enumerate(names):
                row[f"{coordinate}_rms_{name}"] = d["direction_rms"][j]
                row[f"{coordinate}_eigenvalue_{j}"] = d["eigenvalues"][j]
                row[f"{coordinate}_unique_weak_{name}"] = d["unique_weak_vector"][j] if d["unique_weak_vector"] is not None else None
                for q in range(j+1, len(names)):
                    row[f"{coordinate}_correlation_{name}__{names[q]}"] = d["correlation"][j][q]
        vector = point["scaled"]["unique_weak_vector"]
        for j, name in enumerate(names):
            row[f"physical_weak_from_scaled_{name}"] = scales[j]*vector[j] if vector is not None else None
        rows.append(row)
    return rows


def summarize(field, images):
    scaled = [d["scaled"] for d in field["diagnostics"]]
    sweep = np.asarray([s["ranks"] for s in field["rank_sweep"]])
    conditions = [d["condition_number"] for d in scaled if d["condition_number"] is not None]
    return {"images": images[0]["images"], "points": len(scaled),
            "local_rank_min": min(d["rank"] for d in scaled), "local_rank_max": max(d["rank"] for d in scaled),
            "global_rank_min": min(d["global_reference_rank"] for d in scaled),
            "global_rank_max": max(d["global_reference_rank"] for d in scaled),
            "rank_sweep_sensitive_points": int(np.count_nonzero(sweep.max(0) != sweep.min(0))),
            "nonunique_weak_space_points": sum(d["weak_subspace_dimension"] > 1 for d in scaled),
            "max_condition_finite": max(conditions) if conditions else None,
            "roundoff_negative_scaled_eigenvalues": sum(d["negative_roundoff_eigenvalues"] for d in scaled),
            "reference_point_index": field["reference_point_index"], "reference_point": field["reference_point"],
            "max_neighbor_relative_change": max((e["relative_change"] for e in field["edges"]), default=None)}


def analyze(run_path, output, settings, operators=None, splits=None, controls_path=None, refresh=False):
    root, run = load_run(run_path)
    plan = run["plan"]
    if plan["stage"] not in ("pilot", "full"):
        raise ValueError("--run must be a Task 3 pilot/full image-metric run; use --controls-run for controls.")
    names = list(plan["grids"]) if operators is None else operators
    split_names = list(plan["selections"]) if splits is None else splits
    if (not names or len(set(names)) != len(names) or any(n not in plan["grids"] or not safe_name(n) for n in names)
            or not split_names or len(set(split_names)) != len(split_names)
            or any(s not in plan["selections"] or s not in ("train", "valid") for s in split_names)):
        raise ValueError("Select unique operators and splits present in the source run.")
    data = {(name, split): load_matrices(root, run, name, split) for name in names for split in split_names}
    controls = load_controls(controls_path) if controls_path else None
    output = Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Task 4 output must be separate from the Task 3 source tree.")
    if controls and (output == controls[0] or controls[0] in output.parents or output in controls[0].parents):
        raise ValueError("Task 4 output must be separate from the controls source tree.")
    identity_data = {
        "source_run_sha256": sha256(root/"run.json"), "source_fingerprint": run["fingerprint"],
        "inputs": {f"{name}/{split}": item["source_sha256"] for (name, split), item in data.items()},
        "controls_run_sha256": sha256(controls[0]/"run.json") if controls else None,
        "controls_sha256": controls[3] if controls else None,
        "settings": settings, "operators": names, "splits": split_names,
        "code_hashes": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": {"numpy": np.__version__, "python": platform.python_version()}}
    identity = fingerprint(identity_data)
    if output.exists() and any(output.iterdir()):
        if not refresh or not (output/"analysis_run.json").is_file():
            raise ValueError("Output exists. Choose a new directory or --refresh for the same analysis identity.")
        if read_json(output/"analysis_run.json").get("fingerprint") != identity:
            raise ValueError("Refresh identity changed. Use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"analysis_run.json", {"schema_version": 1, "fingerprint": identity, "plan": identity_data})
    summary = {"schema_version": 1, "fingerprint": identity, "status": "ANALYSIS_COMPLETE",
               "source_stage": plan["stage"], "source_fingerprint": run["fingerprint"],
               "operators": {}, "split_comparisons": {}, "controls": None,
               "scope": "Descriptive analysis of the supplied selected-image matrices and sampled points; not a statistical or geometry acceptance decision."}
    fields = {}
    for (name, split), item in data.items():
        grid, arrays = item["grid"], item["arrays"]
        field = analyze_field(arrays["points"], arrays["mean_metric"], arrays["scales"], settings)
        images = describe_images(arrays["per_image_metric"], field, arrays["scales"], settings)
        destination = output/name/split
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination/"field_analysis.json", {"grid": grid, "field": field, "image_descriptions": images})
        write_csv(destination/"point_analysis.csv", flatten_points(grid, field, images))
        write_csv(destination/"neighbor_analysis.csv", field["edges"])
        sweep_rows = [{"atol": row["atol"], "rtol": row["rtol"], "point_index": p, "rank": rank}
                      for row in field["rank_sweep"] for p, rank in enumerate(row["ranks"])]
        write_csv(destination/"rank_sensitivity.csv", sweep_rows)
        summary["operators"].setdefault(name, {})[split] = summarize(field, images)
        fields[name, split] = field
        if settings["plots"]:
            from .plots import plot_field
            plot_field(destination, grid, field, name, split)
        print(f"Analysed {name}/{split}: {len(arrays['image_ids'])} images, {len(arrays['points'])} points.", flush=True)
    if "train" in split_names and "valid" in split_names:
        for name in names:
            grid = data[name, "train"]["grid"]
            s = np.asarray(grid["scales"])
            a = data[name, "train"]["arrays"]["mean_metric"]*s[None, :, None]*s[None, None, :]
            b = data[name, "valid"]["arrays"]["mean_metric"]*s[None, :, None]*s[None, None, :]
            rows = [{"point_index": i, "scaled_frobenius_difference": symmetric_relative_change(x, y)[0],
                     "scaled_relative_difference": symmetric_relative_change(x, y)[1],
                     "local_rank_train": fields[name, "train"]["diagnostics"][i]["scaled"]["rank"],
                     "local_rank_valid": fields[name, "valid"]["diagnostics"][i]["scaled"]["rank"],
                     **subspace_change(fields[name, "train"]["diagnostics"][i]["scaled"],
                                       fields[name, "valid"]["diagnostics"][i]["scaled"])} for i, (x, y) in enumerate(zip(a, b))]
            write_csv(output/name/"split_comparison.csv", rows)
            summary["split_comparisons"][name] = {
                "points": len(rows), "max_relative_difference": max(r["scaled_relative_difference"] for r in rows),
                "comparable_weak_space_points": sum(r["comparable_weak_spaces"] for r in rows)}
    if controls:
        count = 0
        control_summary = []
        for (name, pattern), rows in sorted(controls[2].items()):
            points = [r["point"] for r in rows]
            scales = rows[0]["diagnostics"]["scales"]
            if any(r["diagnostics"]["scales"] != scales for r in rows):
                raise ValueError("Inconsistent control characteristic scales.")
            field = analyze_field(points, [r["metric"] for r in rows], scales, settings)
            count += len(rows)
            control_summary.append({"operator": name, "pattern": pattern, "cases": len(rows),
                                    "field": field, "max_analytic_gram_error": max(r["max_abs_gram_error"] for r in rows)})
        atomic_json(output/"controls_analysis.json", control_summary)
        summary["controls"] = {"groups": len(control_summary), "cases": count}
    atomic_json(output/"summary.json", summary)
    write_report(output/"REPORT.md", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Task 3 pilot/full or synthetic image-run directory")
    parser.add_argument("--controls-run", help="Optional Task 3 controls directory")
    parser.add_argument("--config", default="configs/task04.yaml")
    parser.add_argument("--output", help="Separate Task 4 destination (default outputs/task04/<source name>)")
    parser.add_argument("--operators", nargs="+")
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.no_plots:
        settings["plots"] = False
    output = args.output or str(Path("outputs/task04")/Path(args.run).resolve().name)
    analyze(args.run, output, settings, args.operators, args.splits, args.controls_run, args.refresh)
    print(f"Task 4 outputs: {Path(output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
