"""Export scientific plots while making zero directions and degeneracies explicit."""
from pathlib import Path
import numpy as np


def plot_field(destination, grid, field, operator, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    destination = Path(destination)
    k, points = len(grid["scales"]), np.asarray(grid["points"])
    x = points[:, 0] if k == 1 else np.arange(len(points))
    xlabel = grid["parameter_names"][0] if k == 1 else "Parameter point index (see point_analysis.csv)"

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(destination/f"{name}.{extension}", dpi=160)
        plt.close(fig)

    fig, panels = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for panel, coordinate in zip(panels[:2], ("raw", "scaled")):
        eigenvalues = np.asarray([d[coordinate]["eigenvalues"] for d in field["diagnostics"]])
        positive = eigenvalues[eigenvalues > 0]
        for j in range(k):
            panel.plot(x, np.where(eigenvalues[:, j] > 0, eigenvalues[:, j], np.nan), ".-", label=f"eigenvalue {j}")
        panel.set_yscale("log")
        if not positive.size:
            panel.set_ylim(1e-16, 1.)
        panel.set_title(f"{coordinate.capitalize()} eigenvalues\n{np.count_nonzero(eigenvalues <= 0)} nonpositive values omitted from log plot", fontsize=10)
        panel.legend(fontsize=8)
    local = [d["scaled"]["rank"] for d in field["diagnostics"]]
    global_ = [d["scaled"]["global_reference_rank"] for d in field["diagnostics"]]
    panels[2].plot(x, local, "o-", label="local threshold")
    panels[2].plot(x, global_, "x--", label="grid-wide reference")
    panels[2].set_yticks(range(k+1))
    panels[2].set_ylim(-.2, k+.2)
    panels[2].set_title("Scaled numerical rank")
    panels[2].legend(fontsize=8)
    for panel in panels:
        panel.set_xlabel(xlabel)
        panel.grid(alpha=.25)
    fig.suptitle(f"{operator} / {split}: spectra and rank conventions")
    save(fig, "spectra_and_rank")

    rows = field["rank_sweep"]
    fig, panel = plt.subplots(figsize=(max(9, min(15, len(points)*.35)), 5), constrained_layout=True)
    rank_cmap = plt.get_cmap("viridis", k+1)
    norm = BoundaryNorm(np.arange(k+2)-.5, k+1)
    mesh = panel.imshow(np.asarray([r["ranks"] for r in rows]), aspect="auto", interpolation="nearest", cmap=rank_cmap, norm=norm)
    panel.set_yticks(range(len(rows)), labels=[f"atol={r['atol']:.0e}, rtol={r['rtol']:.0e}" for r in rows], fontsize=8)
    panel.set_xlabel("Parameter point index")
    panel.set_title(f"{operator} / {split}: scaled rank threshold sensitivity")
    fig.colorbar(mesh, ax=panel, ticks=range(k+1), label="Numerical rank")
    save(fig, "rank_sensitivity")

    if field["edges"]:
        fig, panels = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for axis, name in enumerate(grid["parameter_names"]):
            rows = [(i, r) for i, r in enumerate(field["edges"]) if r["parameter_index"] == axis]
            panels[0].scatter([i for i, _ in rows], [r["relative_change"] for _, r in rows], label=name, s=18)
            angles = [(i, r) for i, r in rows if r["max_principal_angle_degrees"] is not None]
            panels[1].scatter([i for i, _ in angles], [r["max_principal_angle_degrees"] for _, r in angles], label=name, s=18)
        panels[0].set_ylabel("Relative scaled metric change")
        panels[1].set_ylabel("Maximum principal angle (degrees)")
        panels[1].set_ylim(-2, 92)
        panels[0].set_title("Adjacent axial metric changes")
        panels[1].set_title("Weakest subspace change\nDifferent dimensions are omitted", fontsize=10)
        for panel in panels:
            panel.set_xlabel("Edge index (see neighbor_analysis.csv)")
            panel.legend(fontsize=8)
            panel.grid(alpha=.25)
        fig.suptitle(f"{operator} / {split}: variation in the fixed chart")
        save(fig, "neighbor_variation")

    axes = grid["axes"]
    if k == 2 and axes is not None and all(len(a) > 1 for a in axes):
        diagnostic = [d["scaled"] for d in field["diagnostics"]]
        data = [
            ("Scaled direction RMS: "+grid["parameter_names"][0], [d["direction_rms"][0] for d in diagnostic]),
            ("Scaled direction RMS: "+grid["parameter_names"][1], [d["direction_rms"][1] for d in diagnostic]),
            ("Derivative correlation", [d["correlation"][0][1] for d in diagnostic]),
            ("Smallest / largest scaled eigenvalue", [d["smallest_to_largest_eigenvalue_ratio"] for d in diagnostic]),
            ("Scaled rank: grid-wide reference", [d["global_reference_rank"] for d in diagnostic]),
            ("Weakest eigenspace dimension", [d["weak_subspace_dimension"] for d in diagnostic]),
        ]
        fig, panels = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgrey")
        signed_cmap = plt.get_cmap("coolwarm").copy()
        signed_cmap.set_bad("lightgrey")
        for index, (panel, (title, values)) in enumerate(zip(panels.flat, data)):
            values = np.asarray([np.nan if v is None else v for v in values]).reshape(len(axes[0]), len(axes[1]))
            options = {"cmap": cmap}
            if index in (4, 5):
                options = {"cmap": rank_cmap, "norm": norm}
            elif index == 2:
                options = {"cmap": signed_cmap, "vmin": -1, "vmax": 1}
            mesh = panel.pcolormesh(axes[0], axes[1], values.T, shading="nearest", **options)
            panel.set_xlim(axes[0][0], axes[0][-1])
            panel.set_ylim(axes[1][0], axes[1][-1])
            panel.set_xlabel(grid["parameter_names"][0])
            panel.set_ylabel(grid["parameter_names"][1])
            panel.set_title(title, fontsize=9)
            fig.colorbar(mesh, ax=panel, ticks=range(k+1) if index in (4, 5) else None)
        fig.suptitle(f"{operator} / {split}: identifiability diagnostics at sampled points")
        save(fig, "identifiability_grid")
