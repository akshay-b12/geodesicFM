"""Task 3 inner products and unregularized coordinate diagnostics."""
import numpy as np
import torch


def gram_metric(jacobian):
    """Parameter-first J -> mean over ALL output scalars (RGB and core pixels).

    The normalization is 1/(C*H*W), not a sum, and not 1/(H*W).
    Each realization must be reduced BEFORE averaging realizations.
    """
    if jacobian.dtype != torch.float64 or jacobian.ndim < 2:
        raise ValueError("Metric estimation requires parameter-first float64 Jacobians.")
    if not torch.isfinite(jacobian).all() or jacobian.numel() == 0:
        raise ValueError("Empty or nonfinite Jacobian.")
    flat = jacobian.reshape(jacobian.shape[0], -1)
    return (flat @ flat.T / flat.shape[1]).detach()


def image_average(crop_metrics, image_ids):
    """Equal crops within image, equal images; unequal crop counts are allowed."""
    values = np.asarray(crop_metrics, dtype=np.float64)
    if len(values) != len(image_ids) or not image_ids:
        raise ValueError("Need nonempty, aligned crop metrics and image IDs.")
    ids = sorted(set(image_ids))
    per_image = np.stack([values[[i for i, x in enumerate(image_ids) if x == key]].mean(0)
                          for key in ids])
    return ids, per_image, per_image.mean(0)


def metric_diagnostics(metric, scales, *, rank_atol=1e-12, rank_rtol=1e-8,
                       psd_atol=1e-14, psd_rtol=1e-10):
    """Raw and S G S diagnostics; never replace G by a clamped/jittered matrix.

    If xi = S z, columns of J_z are scaled columns of J_xi. Eigenvectors
    below are unit vectors in their specified coordinates, in ascending
    eigenvalue order. Their signs and bases inside repeated eigenspaces are
    arbitrary. Classification tolerances are saved with the result.
    """
    g = np.asarray(metric, dtype=np.float64)
    s = np.asarray(scales, dtype=np.float64)
    if g.shape != (len(s), len(s)) or not np.isfinite(g).all():
        raise ValueError("Invalid metric shape or nonfinite matrix.")
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("Scales must be finite and positive.")
    tolerances = (rank_atol, rank_rtol, psd_atol, psd_rtol)
    if any(not np.isfinite(v) or v < 0 for v in tolerances):
        raise ValueError("Diagnostic tolerances must be finite and nonnegative.")
    symmetry_error = float(np.max(np.abs(g-g.T)))
    symmetry_limit = psd_atol + psd_rtol * float(np.max(np.abs(g)))
    if symmetry_error > symmetry_limit:
        raise ValueError("Metric is not symmetric within the declared tolerance.")
    result = {"symmetry_error": symmetry_error, "scales": s.tolist()}
    for label, matrix in (("raw", g), ("scaled", s[:, None]*g*s[None, :])):
        # Symmetrize only the eigensolver input; stored metric is untouched.
        eig, vectors = np.linalg.eigh((matrix+matrix.T)/2)
        spectral_size = float(np.max(np.abs(eig)))
        psd_limit = psd_atol + psd_rtol*spectral_size
        if eig[0] < -psd_limit:
            raise ValueError(f"{label} metric is not PSD: eigenvalue {eig[0]}.")
        threshold = rank_atol + rank_rtol*max(float(eig[-1]), 0.0)
        rank = int(np.count_nonzero(eig > threshold))
        full = rank == len(s)
        diagonal = np.diag(matrix)
        correlation = np.full_like(matrix, np.nan)
        # Undefined correlations for zero/roundoff-negative diagonal entries.
        positive = diagonal > 0
        for i in range(len(s)):
            for j in range(len(s)):
                if positive[i] and positive[j]:
                    correlation[i, j] = matrix[i, j]/np.sqrt(diagonal[i]*diagonal[j])
        result[label] = {
            "eigenvalues": eig.tolist(), "eigenvectors_columns": vectors.tolist(),
            "rank": rank, "rank_threshold": threshold, "psd_tolerance": psd_limit,
            "roundoff_negative_eigenvalues": int(np.count_nonzero(eig < 0)),
            "condition_number": float(eig[-1]/eig[0]) if full else None,
            "singular_at_threshold": not full,
            "direction_rms": np.sqrt(np.maximum(diagonal, 0)).tolist(),
            "correlation": [[float(v) if np.isfinite(v) else None for v in row]
                            for row in correlation],
            "weak_direction": vectors[:, 0].tolist(),
        }
    return result


def neighbor_variation(points, metrics, scales):
    """Edges differing in one coordinate by an adjacent sampled value.

    Irregular point sets get only existing axial edges. This is a descriptive
    finite difference in a fixed chart, NOT curvature or a confidence test.
    Relative change uses max endpoint Frobenius norms; a zero-to-zero edge
    has zero change. Step sizes are expressed in characteristic units.
    """
    p, g, s = map(lambda x: np.asarray(x, dtype=np.float64), (points, metrics, scales))
    scaled = g*s[None, :, None]*s[None, None, :]
    rows = []
    for axis in range(p.shape[1]):
        groups = {}
        for i, point in enumerate(p):
            groups.setdefault(tuple(np.delete(point, axis)), []).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: p[i, axis])
            for a, b in zip(indices[:-1], indices[1:]):
                step = float((p[b, axis]-p[a, axis])/s[axis])
                delta = float(np.linalg.norm(scaled[b]-scaled[a]))
                denominator = max(float(np.linalg.norm(scaled[a])), float(np.linalg.norm(scaled[b])))
                rows.append({"point_a": a, "point_b": b, "parameter_index": axis,
                             "scaled_step": step, "scaled_frobenius_change": delta,
                             "relative_change": delta/denominator if denominator else 0.0,
                             "change_per_scaled_step": delta/step})
    return rows
