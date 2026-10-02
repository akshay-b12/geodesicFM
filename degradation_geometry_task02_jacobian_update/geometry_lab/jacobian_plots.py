"""Exported numerical plots; plotting floors do not change numeric gate results."""
from pathlib import Path
import numpy as np
import torch


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_sweeps(destination, rows):
    if not rows:
        return
    plt = _pyplot()
    directory = Path(destination)/"plots"
    directory.mkdir(exist_ok=True)
    for operator in sorted({r["operator"] for r in rows}):
        subset = [r for r in rows if r["operator"] == operator]
        names = list(dict.fromkeys(r["parameter_name"] for r in subset))
        fig,axes = plt.subplots(len(names),2,figsize=(10,3*len(names)),squeeze=False)
        for i,name in enumerate(names):
            selected = [r for r in subset if r["parameter_name"] == name]
            steps = sorted({r["relative_step"] for r in selected})
            for col,key,title in ((0,"error_rms","Absolute RMS error"),(1,"relative_l2","Relative L2 error")):
                med,low,high = [],[],[]
                for step in steps:
                    values = [r[key] for r in selected if r["relative_step"] == step and r[key] is not None]
                    med.append(max(float(np.median(values)),1e-18) if values else np.nan)
                    low.append(max(float(np.min(values)),1e-18) if values else np.nan)
                    high.append(max(float(np.max(values)),1e-18) if values else np.nan)
                ax = axes[i,col]
                ax.loglog(steps,med,"o-",label="median across cases")
                ax.fill_between(steps,low,high,alpha=0.18,label="min-max across cases")
                ax.set_xlabel(r"$\epsilon$ (step $h_i=\epsilon s_i$)")
                ax.set_ylabel(title)
                ax.set_title(f"{name}: {title}",fontsize=10)
                ax.grid(True,which="both",alpha=0.25)
                ax.legend(fontsize=8)
        fig.suptitle(f"{operator}: all cases",fontsize=12)
        fig.text(0.5,0.015,"Relative panels exclude near-zero references. Read CSVs for individual failures.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,0.035,1,0.96))
        fig.savefig(directory/f"sweep_{operator}.png",dpi=150)
        plt.close(fig)


def plot_examples(destination, paths):
    if not paths:
        return
    plt = _pyplot()
    directory = Path(destination)/"plots"
    directory.mkdir(exist_ok=True)
    for path in paths:
        data = torch.load(path,map_location="cpu",weights_only=True)
        ad,fd = data["J_AD"].numpy(),data["J_FD"].numpy()
        names = data["parameter_names"]
        fig,axes = plt.subplots(len(names),3,figsize=(11,3*len(names)),squeeze=False)
        for i,name in enumerate(names):
            limit = max(float(np.nanmax(np.abs(ad[i,0]))),float(np.nanmax(np.abs(fd[i,0]))),1e-12)
            diff = fd[i,0]-ad[i,0]
            error_limit = max(float(np.nanmax(np.abs(diff))),1e-12)
            for col,(image,vmax,title) in enumerate(((ad[i,0],limit,"forward AD"),(fd[i,0],limit,"finite difference"),(diff,error_limit,"FD - AD; own scale"))):
                ax = axes[i,col]
                im = ax.imshow(image,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
                ax.set_title(f"{name}\n{title}",fontsize=10)
                ax.set_axis_off()
                fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        fig.suptitle(f"{data['identity']['operator']}: red-channel derivatives; epsilon={data['example_epsilon']:g}")
        fig.tight_layout()
        fig.savefig(directory/f"{Path(path).stem}.png",dpi=150)
        plt.close(fig)
