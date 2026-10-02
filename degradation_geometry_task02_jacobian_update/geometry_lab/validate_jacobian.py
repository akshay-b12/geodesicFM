"""Task 2 CLI: controls -> DIV2K pilot -> precision -> kernel-support review.

Each run has a fresh named directory. Original Task 0 manifests/config are read
without modification. Machine-readable reports distinguish derivative failures
from representation/support sensitivity and incomplete diagnostic coverage.
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import yaml

from .config import load_config, resolve_path, output_path
from .datasets import ManifestDataset, decode_rgb, rgb_tensor, stable_seed, file_sha256
from .jacobian import (jacobian_jvp, finite_difference_jacobian, derivative_error,
                       adjacent_pass_window, reverse_projection_check)
from .jacobian_cases import (operator_specs, auxiliary_fields, bind_operator,
                            analytic_jacobian, synthetic_context)


def load_settings(path):
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    steps = settings["relative_steps"]
    if len(steps) < 2 or any(not np.isfinite(v) or v <= 0 for v in steps):
        raise ValueError("Provide at least two positive finite relative steps.")
    if any(a <= b for a,b in zip(steps,steps[1:])):
        raise ValueError("relative_steps must be strictly decreasing.")
    if not 2 <= settings["minimum_adjacent_passes"] <= len(steps):
        raise ValueError("Invalid minimum_adjacent_passes.")
    for field in ("noise_realizations","reverse_probes","torch_threads","control_core_size"):
        if type(settings[field]) is not int or settings[field] < 1:
            raise ValueError(f"{field} must be a positive integer.")
    for stage in ("pilot","precision","support"):
        for field in ("max_train_images","max_valid_images"):
            if type(settings[stage][field]) is not int or settings[stage][field] < 0:
                raise ValueError(f"{stage}.{field} must be a nonnegative integer.")
        effective_counts = []
        for split in ("train","valid"):
            explicit = settings[stage].get(f"{split}_image_indices")
            if explicit is not None and (not isinstance(explicit,list) or any(type(v) is not int or v < 0 for v in explicit)
                                         or len(set(explicit)) != len(explicit)):
                raise ValueError("Explicit image indices must be a unique list of nonnegative integers.")
            effective_counts.append(len(explicit) if explicit is not None else settings[stage][f"max_{split}_images"])
        if sum(effective_counts) == 0:
            raise ValueError(f"{stage} must select at least one image.")
        if type(settings[stage]["crops_per_image"]) is not int or settings[stage]["crops_per_image"] < 1:
            raise ValueError("crops_per_image must be positive.")
    for name, tolerance in settings["tolerances"].items():
        for key in ("atol","rtol","signal_floor"):
            if not isinstance(tolerance[key],(int,float)) or not np.isfinite(tolerance[key]) or tolerance[key] < 0:
                raise ValueError(f"Invalid {name}.{key} tolerance.")
    larger = settings["support"]["larger_kernel_size"]
    if type(larger) is not int or larger % 2 == 0 or larger < 3:
        raise ValueError("larger_kernel_size must be an odd integer >= 3.")
    return settings


def dump_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list,tuple)) else value
                             for key,value in row.items()})


def validate_case(fn, xi, spec, settings, *, analytic=None, projection_seed=123):
    """Return full sweep and gates for ONE image/parameter/noise realization."""
    tol = settings["tolerances"][str(xi.dtype).split(".")[-1]]
    reference = jacobian_jvp(fn,xi)
    first, repeated = fn(xi).detach(), fn(xi).detach()
    deterministic = bool(torch.equal(first,repeated))
    reversetol = settings["tolerances"]["reverse64"] if xi.dtype == torch.float64 else tol
    reverse = reverse_projection_check(fn,xi,reference,seed=projection_seed,
               probes=settings["reverse_probes"],tolerances=reversetol)
    analytic_rows = []
    if analytic is not None:
        analytictol = settings["tolerances"]["analytic64"] if xi.dtype == torch.float64 else tol
        for i in range(xi.numel()):
            analytic_rows.append({"coordinate":i,**derivative_error(analytic[i],reference[i],**analytictol)})
    rows, derivatives = [], []
    for epsilon in settings["relative_steps"]:
        steps = [epsilon*scale for scale in spec.scales]
        fd = finite_difference_jacobian(fn,xi,steps,bounds=spec.bounds,boundary="one_sided")
        derivatives.append(fd.jacobian)
        for i,name in enumerate(spec.parameter_names):
            error = derivative_error(reference[i],fd.jacobian[i],**tol)
            rows.append({"coordinate":i,"parameter_name":name,"relative_step":epsilon,
                         "absolute_step":fd.steps[i],"scheme":fd.schemes[i],**error})
    gates = []
    for i,name in enumerate(spec.parameter_names):
        selected = [row for row in rows if row["coordinate"] == i]
        window = adjacent_pass_window([r["passed"] for r in selected],settings["minimum_adjacent_passes"])
        # First qualifying interval is explicitly recorded; all steps stay in CSV.
        window["epsilon_interval"] = ([selected[window["start_index"]]["relative_step"],
                                      selected[window["end_index"]]["relative_step"]] if window["passed"] else None)
        gates.append({"coordinate":i,"parameter_name":name,**window})
    special = []
    if spec.name == "anisotropic_blur" and float(xi[0]) == float(xi[1]):
        zero_error = derivative_error(torch.zeros_like(reference[2]),reference[2],
                                      **(settings["tolerances"]["analytic64"] if xi.dtype == torch.float64 else tol))
        special.append({"check":"equal_axes_zero_theta_derivative",**zero_error})
    passed = deterministic and all(g["passed"] for g in gates) and all(r["passed"] for r in reverse+analytic_rows+special)
    # One common epsilon for example maps. Its status is explicitly recorded.
    common = [all(r["passed"] for r in rows if r["relative_step"] == epsilon)
              for epsilon in settings["relative_steps"]]
    example_idx = next((i for i,p in enumerate(common) if p),len(common)//2)
    return {"passed":passed,"deterministic":deterministic,"gates":gates,
            "reverse":reverse,"analytic":analytic_rows,"special":special,
            "fd_rows":rows,"reference":reference,"example_fd":derivatives[example_idx],
            "example_epsilon":settings["relative_steps"][example_idx],
            "example_fd_all_coordinates_pass":common[example_idx]}


def select_samples(cfg, settings, stage):
    if stage == "controls":
        halo = cfg["sampling"]["halo"]
        core_size = settings["control_core_size"]
        return [{"image_id":f"synthetic/{pattern}","crop_index":0,"halo":halo,
                 "crop_size":core_size,"image":synthetic_context(pattern,core_size,halo),
                 "source":"synthetic_control"} for pattern in settings["control_patterns"]], {}
    samples, manifests = [], {}
    for split,key in (("train","train_hr"),("valid","valid_hr")):
        explicit = settings[stage].get(f"{split}_image_indices")
        count = len(explicit) if explicit is not None else settings[stage][f"max_{split}_images"]
        if count == 0:
            continue
        path = output_path(cfg)/f"{split}_manifest.json"
        override = cfg["dataset"][key]
        ds = ManifestDataset(path,root_override=resolve_path(cfg,override) if override else None,dtype=torch.float64)
        if ds.manifest["sampling"] != cfg["sampling"]:
            raise ValueError("Task 0 sampling config differs from its saved manifest.")
        total = len(ds.manifest["images"])
        if count > total:
            raise ValueError(f"Requested {count} {split} images, manifest contains {total}.")
        n_crops = settings[stage]["crops_per_image"]
        if n_crops > ds.manifest["sampling"]["crops_per_image"]:
            raise ValueError("Requested more crops than Task 0 prepared.")
        indices = set(explicit) if explicit is not None else set(np.linspace(0,total-1,count,dtype=int).tolist())
        if any(index >= total for index in indices):
            raise ValueError(f"Explicit {split} image index lies outside its manifest.")
        for index,rec in enumerate(ds.records):
            if rec["image_index"] in indices and rec["crop_index"] < n_crops:
                item = ds[index]
                item["source"] = split
                item["record"] = rec
                item["root"] = str(ds.root)
                samples.append(item)
        manifests[split] = {"path":str(path),"sha256":file_sha256(path),
                            "selected_image_indices":sorted(indices),"root":str(ds.root)}
    return samples, manifests


def expanded_context(sample, halo, color_space):
    """Larger support uses ORIGINAL-image context around the IDENTICAL core."""
    rec = sample["record"]
    array = decode_rgb(Path(sample["root"])/rec["file_name"])
    t,l,size = rec["top"],rec["left"],rec["crop_size"]
    height,width = array.shape[:2]
    if min(t,l) < halo or t+size+halo > height or l+size+halo > width:
        return None
    return rgb_tensor(array[t-halo:t+size+halo,l-halo:l+size+halo],color_space,dtype=torch.float64)


def run_validation(cfg, settings, *, stage, run_name=None):
    if stage not in ("controls","pilot","precision","support"):
        raise ValueError("Unknown stage.")
    specs_all = operator_specs(cfg["operators"]["sigma_domain"])
    if not cfg["operators"]["sigma_domain"][0] < cfg["operators"]["sigma_domain"][1]:
        raise ValueError("Jacobian validation requires a nonempty blur-width interval.")
    names = settings["operators"]
    if not names or len(set(names)) != len(names) or any(n not in specs_all for n in names):
        raise ValueError("Operator list must be nonempty, unique and drawn from the registry.")
    specs = [specs_all[n] for n in names]
    if stage == "support":
        specs = [spec for spec in specs if spec.has_blur]
        if not specs:
            raise ValueError("Support review requires at least one blur operator.")
        if settings["support"]["larger_kernel_size"] <= cfg["operators"]["kernel_size"]:
            raise ValueError("Support comparison requires a larger kernel than Task 1.")
    name = run_name or stage
    if Path(name).name != name or name in (".","..",""):
        raise ValueError("run_name must be one directory name.")
    destination = resolve_path(cfg,settings["output_dir"])/name
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}. Use --run-name for a fresh run.")
    if settings["plots"]:
        try:
            import matplotlib  # Check before expensive computation or output creation.
        except ImportError as exc:
            raise ImportError("Install requirements_task02.txt or pass --no-plots.") from exc
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(settings["device"])
    if device.type not in ("cpu","cuda"):
        raise ValueError("Use cpu or cuda; other backends have not been validated.")
    samples, manifests = select_samples(cfg,settings,stage)
    if not samples:
        raise ValueError("No samples selected.")
    destination.mkdir(parents=True)
    metadata = {"stage":stage,"run_name":name,"python":platform.python_version(),
                "platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,
                "color_space":cfg["sampling"]["color_space"],"base_kernel_size":cfg["operators"]["kernel_size"],
                "jacobian_layout":"parameter_first: (k,C,H,W)","manifests":manifests,
                "base_config":{k:v for k,v in cfg.items() if not k.startswith("_")},
                "settings":settings,"operator_specs":[asdict(spec) for spec in specs],
                "source_sha256":{str(path.relative_to(Path(__file__).parent)):file_sha256(path)
                                  for path in sorted(Path(__file__).parent.rglob("*.py"))},
                "note":"This run verifies derivatives; it is not metric, curvature or restoration evaluation."}
    dump_json(destination/"run_metadata.json",metadata)
    fd_rows,reverse_rows,analytic_rows,case_rows,comparison_rows,unavailable,example_paths = [],[],[],[],[],[],[]
    examples_saved = set()
    started = time.monotonic()
    print(f"{stage}: {len(samples)} patches; {len(specs)} operator families; device={device}",flush=True)
    for sample_index,sample in enumerate(samples):
        print(f"[{sample_index+1}/{len(samples)}] {sample['image_id']} crop={sample['crop_index']}",flush=True)
        for spec in specs:
            n_noise = settings["noise_realizations"] if spec.has_noise else 1
            print(f"  {spec.name}: {len(spec.points)} points x {n_noise} realizations",flush=True)
            for realization in range(n_noise):
                seed = stable_seed(settings["noise_seed"],sample["image_id"],sample["crop_index"],realization)
                halo = sample["halo"]
                x = sample["image"].to(device)
                if stage == "support":
                    halo = max(halo,settings["support"]["larger_kernel_size"]//2)
                    larger = expanded_context(sample,halo,cfg["sampling"]["color_space"])
                    if larger is None:
                        unavailable.append({"image_id":sample["image_id"],"crop_index":sample["crop_index"],
                                            "operator":spec.name,"realization":realization,
                                            "reason":"Original source image cannot supply larger physical halo."})
                        continue
                    x = larger.to(device)
                fields = auxiliary_fields(x,seed)
                fn64 = bind_operator(spec,x,halo=halo,core_size=sample["crop_size"],
                                     kernel_size=cfg["operators"]["kernel_size"],fields=fields)
                for point_index,values in enumerate(spec.points):
                    identity = {"image_id":sample["image_id"],"crop_index":sample["crop_index"],
                                "operator":spec.name,"point_index":point_index,
                                "parameters":values,"realization":realization,"noise_seed":seed if spec.has_noise else None}
                    try:
                        p64 = torch.tensor(values,dtype=torch.float64,device=device)
                        if stage in ("controls","pilot"):
                            analytic = analytic_jacobian(spec,x,p64,halo=halo,core_size=sample["crop_size"],fields=fields,
                                                         kernel_size=cfg["operators"]["kernel_size"])
                            result = validate_case(fn64,p64,spec,settings,analytic=analytic,
                                    projection_seed=stable_seed(settings["projection_seed"],spec.name,point_index))
                            row = {**identity,"dtype":"float64","passed":result["passed"],"deterministic":result["deterministic"],
                                   "coordinate_gates":result["gates"],"special_checks":result["special"]}
                            case_rows.append(row)
                            fd_rows.extend({**identity,"dtype":"float64",**r} for r in result["fd_rows"])
                            reverse_rows.extend({**identity,"dtype":"float64",**r} for r in result["reverse"])
                            analytic_rows.extend({**identity,"dtype":"float64",**r} for r in result["analytic"])
                            if point_index == 1 and spec.name not in examples_saved and settings["save_example_tensors"]:
                                path = destination/f"example_{spec.name}.pt"
                                torch.save({"J_AD":result["reference"].cpu(),"J_FD":result["example_fd"].cpu(),
                                    "xi":p64.cpu(),"parameter_names":list(spec.parameter_names),"parameter_scales":list(spec.scales),
                                    "example_epsilon":result["example_epsilon"],
                                    "example_fd_all_coordinates_pass":result["example_fd_all_coordinates_pass"],
                                    "identity":identity,"context":x.cpu(),"halo":halo,"core_size":sample["crop_size"],
                                    "fields":{k:v.cpu() for k,v in fields.items()}},path)
                                example_paths.append(path)
                                examples_saved.add(spec.name)
                        elif stage == "precision":
                            master = jacobian_jvp(fn64,p64)
                            x32 = x.float()
                            fields32 = {k:v.float() for k,v in fields.items()}
                            fn32 = bind_operator(spec,x32,halo=halo,core_size=sample["crop_size"],
                                      kernel_size=cfg["operators"]["kernel_size"],fields=fields32)
                            p32 = p64.float()
                            analytic32 = analytic_jacobian(spec,x32,p32,halo=halo,core_size=sample["crop_size"],fields=fields32,
                                                           kernel_size=cfg["operators"]["kernel_size"])
                            result = validate_case(fn32,p32,spec,settings,analytic=analytic32,
                                     projection_seed=stable_seed(settings["projection_seed"],spec.name,point_index))
                            case_rows.append({**identity,"dtype":"float32","passed":result["passed"],
                                              "deterministic":result["deterministic"],"coordinate_gates":result["gates"],
                                              "special_checks":result["special"]})
                            fd_rows.extend({**identity,"dtype":"float32",**r} for r in result["fd_rows"])
                            reverse_rows.extend({**identity,"dtype":"float32",**r} for r in result["reverse"])
                            analytic_rows.extend({**identity,"dtype":"float32",**r} for r in result["analytic"])
                            for i,param in enumerate(spec.parameter_names):
                                comparison_rows.append({**identity,"coordinate":i,"parameter_name":param,"comparison":"float32_vs_float64_AD",
                                    **derivative_error(master[i],result["reference"][i],**settings["tolerances"]["precision"])})
                        else:
                            fnlarge = bind_operator(spec,x,halo=halo,core_size=sample["crop_size"],
                                       kernel_size=settings["support"]["larger_kernel_size"],fields=fields)
                            base,large = jacobian_jvp(fn64,p64),jacobian_jvp(fnlarge,p64)
                            for i,param in enumerate(spec.parameter_names):
                                comparison_rows.append({**identity,"coordinate":i,"parameter_name":param,
                                    "comparison":"base_vs_larger_kernel_AD","base_kernel":cfg["operators"]["kernel_size"],
                                    "larger_kernel":settings["support"]["larger_kernel_size"],
                                    **derivative_error(large[i],base[i],**settings["tolerances"]["support"])})
                    except Exception as exc:
                        case_rows.append({**identity,"passed":False,"error_type":type(exc).__name__,"error":str(exc)})
                        print(f"ERROR {spec.name} point {point_index}: {exc}",flush=True)
    missing_ops = sorted(set(spec.name for spec in specs)-set(r["operator"] for r in (comparison_rows if stage == "support" else case_rows)))
    case_failures = [r for r in case_rows if not r["passed"]]
    comparison_failures = [r for r in comparison_rows if not r["passed"]]
    if case_failures:
        status = "FAIL" if stage in ("controls","pilot") else "REVIEW_REQUIRED"
    elif missing_ops or (stage == "support" and not comparison_rows):
        status = "INCOMPLETE"
    elif comparison_failures:
        status = "REVIEW_REQUIRED"
    elif unavailable:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    summary = {"stage":stage,"status":status,"sample_patches":len(samples),
               "cases":len(case_rows),"case_failures":len(case_failures),
               "comparison_rows":len(comparison_rows),"comparison_failures":len(comparison_failures),
               "fd_rows":len(fd_rows),"unavailable_support_entries":len(unavailable),
               "missing_operator_coverage":missing_ops,"elapsed_seconds":time.monotonic()-started,
               "scope":"Only the listed patches, parameter points, realizations, settings and device.",
               "next_action":("Inspect case_summary.json and sweep CSVs before accepting derivatives." if stage in ("controls","pilot")
                   else "Review precision/support sensitivity; float64 remains the geometry baseline.")}
    write_csv(destination/"finite_difference_sweep.csv",fd_rows)
    write_csv(destination/"reverse_projection_checks.csv",reverse_rows)
    write_csv(destination/"analytic_checks.csv",analytic_rows)
    write_csv(destination/"comparison_checks.csv",comparison_rows)
    dump_json(destination/"case_summary.json",case_rows)
    dump_json(destination/"unavailable_support.json",unavailable)
    dump_json(destination/"summary.json",summary)
    if settings["plots"]:
        from .jacobian_plots import plot_sweeps, plot_examples
        plot_sweeps(destination,fd_rows)
        plot_examples(destination,example_paths)
    (destination/"REPORT.md").write_text(
        f"# Jacobian verification: {stage}\n\nStatus: **{status}**\n\n"
        f"Patches: {len(samples)}. Cases: {len(case_rows)}. Case failures: {len(case_failures)}. "
        f"Comparison failures: {len(comparison_failures)}.\n\n"
        "A PASS verifies only the documented numerical cases. It does not establish metric stability, "
        "curvature, blind inferability or restoration benefit. Precision/support REVIEW_REQUIRED is "
        "separate from a float64 derivative-correctness failure.\n\n"
        "Read summary.json, case_summary.json and the per-coordinate CSVs; no failure is averaged away. "
        "Unavailable larger-halo entries are recorded in unavailable_support.json.\n",
        encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Report: {destination}",flush=True)
    return summary,destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--settings",default="configs/jacobian.yaml")
    parser.add_argument("--stage",choices=("controls","pilot","precision","support"),required=True)
    parser.add_argument("--run-name",help="Fresh named output folder; defaults to stage name")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--operators",help="Comma-separated subset of registered operator names")
    args = parser.parse_args()
    cfg,settings = load_config(args.config),load_settings(args.settings)
    if args.no_plots:
        settings["plots"] = False
    if args.operators:
        settings["operators"] = [s.strip() for s in args.operators.split(",")]
    summary,_ = run_validation(cfg,settings,stage=args.stage,run_name=args.run_name)
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
