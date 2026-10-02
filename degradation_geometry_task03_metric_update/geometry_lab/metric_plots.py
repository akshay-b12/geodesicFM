"""Scientific plots from saved means. Nonpositive eigenvalues remain in data."""
import math
from pathlib import Path
import numpy as np


def plot_metric(destination, grid, mean, diagnostics, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    destination = Path(destination)
    points = np.asarray(grid["points"])
    names, axes = grid["parameter_names"], grid["axes"]
    k = len(names)
    eigenvalues = np.asarray([d["scaled"]["eigenvalues"] for d in diagnostics])
    ranks = np.asarray([d["scaled"]["rank"] for d in diagnostics])
    conditions = np.asarray([d["scaled"]["condition_number"]
                             if d["scaled"]["condition_number"] is not None else np.nan for d in diagnostics])
    x = points[:, 0] if k == 1 else np.arange(len(points))
    xlabel = names[0] if k == 1 else "Parameter point index (see point_summary.csv)"
    fig, panels = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for i in range(k):
        for j in range(i, k):
            panels[0].plot(x, mean[:, i, j], ".-", label=f"G[{names[i]},{names[j]}]")
    for j in range(k):
        panels[1].plot(x, eigenvalues[:, j], ".-", label=f"eigenvalue {j}")
    panels[2].plot(x, ranks, "o-", color="tab:purple")
    panels[2].set_yticks(range(k+1))
    panels[2].set_ylim(-0.2, k+0.2)
    for panel, title in zip(panels, ("Raw metric entries", "Scaled eigenvalues (ascending)", "Scaled numerical rank")):
        panel.set_title(title)
        panel.set_xlabel(xlabel)
        panel.grid(alpha=0.25)
    panels[0].legend(fontsize=7)
    panels[1].legend(fontsize=8)
    fig.suptitle(f"{destination.parent.name} / {split}: image-averaged metric")
    for extension in ("png", "pdf"):
        fig.savefig(destination/f"metric_overview.{extension}", dpi=160)
    plt.close(fig)
    if k == 2 and axes is not None and all(len(a) > 1 for a in axes):
        components = [(f"Raw G[{names[i]},{names[j]}]", mean[:, i, j])
                      for i in range(k) for j in range(i, k)]
        components += [("Scaled smallest eigenvalue", eigenvalues[:, 0]),
                       ("Scaled numerical rank", ranks),
                       ("log10 scaled condition (grey = singular)", np.log10(conditions))]
        fig, panels = plt.subplots(math.ceil(len(components)/3), 3, figsize=(15, 8), constrained_layout=True)
        colormap = plt.get_cmap("viridis").copy()
        colormap.set_bad("lightgrey")
        for panel, (title, values) in zip(panels.flat, components):
            values = values.reshape(len(axes[0]), len(axes[1]))
            rank_panel = title == "Scaled numerical rank"
            rank_cmap = plt.get_cmap("viridis", k+1)
            options = {"cmap": rank_cmap, "norm": BoundaryNorm(np.arange(k+2)-.5, k+1)} if rank_panel else {"cmap": colormap}
            mesh = panel.pcolormesh(axes[0], axes[1], values.T, shading="nearest", **options)
            panel.set_title(title, fontsize=9)
            panel.set_xlabel(names[0])
            panel.set_ylabel(names[1])
            panel.set_xlim(axes[0][0], axes[0][-1])
            panel.set_ylim(axes[1][0], axes[1][-1])
            fig.colorbar(mesh, ax=panel, ticks=range(k+1) if rank_panel else None)
        fig.suptitle(f"{destination.parent.name} / {split}: sampled parameter grid")
        for extension in ("png", "pdf"):
            fig.savefig(destination/f"metric_grid.{extension}", dpi=160)
        plt.close(fig)
