"""Unregularized metric/eigen analysis in explicitly stated coordinates."""
import itertools
import numpy as np


def inspect_matrix(matrix, settings):
    g = np.asarray(matrix, dtype=np.float64)
    if g.ndim != 2 or not len(g) or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError("Need a finite square metric matrix.")
    symmetry_error = float(np.max(np.abs(g-g.T)))
    limit = settings["psd_atol"]+settings["psd_rtol"]*float(np.max(np.abs(g)))
    if symmetry_error > limit:
        raise ValueError("Metric asymmetry exceeds the declared tolerance.")
    # Only eigensolver input is symmetrized. The original matrix is untouched.
    eigenvalues, eigenvectors = np.linalg.eigh((g+g.T)/2)
    psd_limit = settings["psd_atol"]+settings["psd_rtol"]*float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -psd_limit:
        raise ValueError("Metric has a negative eigenvalue beyond the PSD tolerance.")
    maximum = max(float(eigenvalues[-1]), 0.)
    threshold = settings["rank_atol"]+settings["rank_rtol"]*maximum
    rank = int(np.count_nonzero(eigenvalues > threshold))
    gap_tolerance = settings["eigengap_atol"]+settings["eigengap_rtol"]*maximum
    weak_size = int(np.count_nonzero(eigenvalues-eigenvalues[0] <= gap_tolerance))
    weak_basis = eigenvectors[:, :weak_size]
    weak_vector = None
    if weak_size == 1:
        vector = weak_basis[:, 0].copy()
        # A display convention only. Subspace comparisons do not use this sign.
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        weak_vector = vector.tolist()
    positive = np.maximum(eigenvalues, 0.)  # Fractions only; not stored eigenvalues.
    total = float(positive.sum())
    fractions = (positive/total).tolist() if total else None
    diagonal = np.diag(g)
    correlation = np.full_like(g, np.nan)
    for i in range(len(g)):
        for j in range(len(g)):
            if diagonal[i] > 0 and diagonal[j] > 0:
                correlation[i, j] = g[i, j]/(np.sqrt(diagonal[i])*np.sqrt(diagonal[j]))
    pairs = [float(correlation[i, j]) for i in range(len(g)) for j in range(i+1, len(g))
             if np.isfinite(correlation[i, j])]
    return {
        "eigenvalues": eigenvalues.tolist(), "eigenvectors_columns": eigenvectors.tolist(),
        "symmetry_error": symmetry_error, "psd_tolerance": psd_limit,
        "negative_roundoff_eigenvalues": int(np.count_nonzero(eigenvalues < 0)),
        "rank": rank, "rank_threshold": threshold,
        "condition_number": float(eigenvalues[-1]/eigenvalues[0]) if rank == len(g) else None,
        "smallest_to_largest_eigenvalue_ratio": float(eigenvalues[0]/maximum) if maximum else None,
        "direction_rms": np.sqrt(np.maximum(diagonal, 0.)).tolist(),
        "correlation": [[float(v) if np.isfinite(v) else None for v in row] for row in correlation],
        "max_abs_offdiagonal_correlation": max(map(abs, pairs)) if pairs else None,
        "positive_eigenvalue_fractions": fractions,
        "participation_ratio": 1./float((positive/total)@(positive/total)) if total else None,
        "weak_subspace_dimension": weak_size, "weak_basis_columns": weak_basis.tolist(),
        "eigengap_tolerance": gap_tolerance, "unique_weak_vector": weak_vector,
    }


def subspace_change(a, b):
    """Equal-dimensional weakest spaces: sign/basis-invariant principal angles."""
    va = np.asarray(a["weak_basis_columns"])
    vb = np.asarray(b["weak_basis_columns"])
    if va.shape[1] != vb.shape[1]:
        return {"comparable_weak_spaces": False, "max_principal_angle_degrees": None,
                "normalized_projector_distance": None}
    cosine = np.clip(np.linalg.svd(va.T@vb, compute_uv=False), 0., 1.)
    delta = np.linalg.norm(va@va.T-vb@vb.T)/np.sqrt(2*va.shape[1])
    return {"comparable_weak_spaces": True,
            "max_principal_angle_degrees": float(np.degrees(np.arccos(cosine.min()))),
            "normalized_projector_distance": float(delta)}


def axial_edges(points):
    p = np.asarray(points)
    rows = []
    for axis in range(p.shape[1]):
        groups = {}
        for i, point in enumerate(p):
            groups.setdefault(tuple(np.delete(point, axis)), []).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: p[i, axis])
            rows.extend((a, b, axis) for a, b in zip(indices[:-1], indices[1:]))
    return rows


def symmetric_relative_change(a, b):
    absolute = float(np.linalg.norm(a-b))
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    return absolute, absolute/denominator if denominator else 0.


def analyze_field(points, matrices, scales, settings):
    p, raw, s = (np.asarray(v, dtype=np.float64) for v in (points, matrices, scales))
    k = len(s)
    if (p.ndim != 2 or p.shape[1] != k or raw.shape != (len(p), k, k)
            or not len(p) or not np.isfinite(p).all() or not np.isfinite(s).all()
            or np.any(s <= 0) or len(np.unique(p, axis=0)) != len(p)):
        raise ValueError("Invalid field dimensions, points or characteristic scales.")
    scaled = raw*s[None, :, None]*s[None, None, :]
    diagnostics = [{"point_index": i, "point": point.tolist(),
                    "raw": inspect_matrix(raw[i], settings), "scaled": inspect_matrix(scaled[i], settings)}
                   for i, point in enumerate(p)]
    globals_ = {coordinate: max(max(d[coordinate]["eigenvalues"][-1], 0.) for d in diagnostics)
                for coordinate in ("raw", "scaled")}
    for d in diagnostics:
        for coordinate in ("raw", "scaled"):
            threshold = settings["rank_atol"]+settings["rank_rtol"]*globals_[coordinate]
            d[coordinate]["global_rank_threshold"] = threshold
            d[coordinate]["global_reference_rank"] = int(np.count_nonzero(np.asarray(d[coordinate]["eigenvalues"]) > threshold))
    reference = int(np.argmin(np.linalg.norm((p-np.median(p, axis=0))/s, axis=1)))
    for i, d in enumerate(diagnostics):
        absolute, relative = symmetric_relative_change(scaled[reference], scaled[i])
        d["reference_absolute_change"] = absolute
        d["reference_relative_change"] = relative
    edges = []
    for a, b, axis in axial_edges(p):
        step = float((p[b, axis]-p[a, axis])/s[axis])
        absolute, relative = symmetric_relative_change(scaled[a], scaled[b])
        edges.append({"point_a": a, "point_b": b, "parameter_index": axis,
                      "scaled_step": step, "scaled_frobenius_change": absolute,
                      "relative_change": relative, "change_per_scaled_step": absolute/step,
                      "scaled_local_rank_changed": diagnostics[a]["scaled"]["rank"] != diagnostics[b]["scaled"]["rank"],
                      **subspace_change(diagnostics[a]["scaled"], diagnostics[b]["scaled"])})
    sweep = []
    eigenvalues = np.asarray([d["scaled"]["eigenvalues"] for d in diagnostics])
    for atol, rtol in itertools.product(settings["rank_sweep_atol"], settings["rank_sweep_rtol"]):
        thresholds = atol+rtol*np.maximum(eigenvalues[:, -1], 0.)
        sweep.append({"atol": atol, "rtol": rtol,
                      "ranks": np.sum(eigenvalues > thresholds[:, None], axis=1).tolist()})
    return {"diagnostics": diagnostics, "edges": edges, "rank_sweep": sweep,
            "reference_point_index": reference, "reference_point": p[reference].tolist(),
            "global_max_eigenvalues": globals_}


def describe_images(per_image, field, scales, settings):
    """Image spread and weakest-space agreement, without bootstrap inference."""
    values = np.asarray(per_image, dtype=np.float64)
    s = np.asarray(scales)
    mean = values.mean(0)
    mean_scaled = mean*s[None, :, None]*s[None, None, :]
    scaled = values*s[None, None, :, None]*s[None, None, None, :]
    rows = []
    for point in range(values.shape[1]):
        raw_results = [inspect_matrix(g, settings) for g in values[:, point]]
        image_results = [inspect_matrix(g, settings) for g in scaled[:, point]]
        reference = field["diagnostics"][point]["scaled"]
        changes = [subspace_change(reference, d) for d in image_results]
        angles = [d["max_principal_angle_degrees"] for d in changes if d["comparable_weak_spaces"]]
        numerator = np.linalg.norm(scaled[:, point]-mean_scaled[point], axis=(1, 2))
        denominator = float(np.linalg.norm(mean_scaled[point]))
        spread = numerator/denominator if denominator else None
        std = values[:, point].std(0, ddof=1) if len(values) > 1 else None
        rows.append({"point_index": point, "images": len(values),
                     "raw_image_ranks": [d["rank"] for d in raw_results],
                     "scaled_image_ranks": [d["rank"] for d in image_results],
                     "scaled_image_rank_min": min(d["rank"] for d in image_results),
                     "scaled_image_rank_max": max(d["rank"] for d in image_results),
                     "scaled_image_full_rank_fraction": float(np.mean([d["rank"] == len(s) for d in image_results])),
                     "mean_relative_image_deviation": float(spread.mean()) if spread is not None else None,
                     "max_relative_image_deviation": float(spread.max()) if spread is not None else None,
                     "comparable_weak_space_images": len(angles),
                     "max_image_weak_space_angle_degrees": max(angles) if angles else None,
                     "image_std_raw_entries": std.tolist() if std is not None else None})
    return rows
