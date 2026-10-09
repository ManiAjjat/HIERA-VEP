from pathlib import Path
import json, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

COLORS={"navy":"#17324D","blue":"#2878B5","cyan":"#4DBBD5","orange":"#E18727","red":"#C44E52","green":"#2E8B57","purple":"#7E57C2","grey":"#7A7A7A","light":"#E8EDF2","black":"#202124"}
def setup():
    mpl.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":9,"axes.titleweight":"bold","axes.linewidth":.7,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"figure.dpi":120,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"axes.facecolor":"white","figure.facecolor":"white"})
def panel(ax,label,title=None):
    ax.text(-.12,1.08,label,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
    if title: ax.set_title(title,loc="left",pad=8)
def find_table(folder,required):
    folder=Path(folder)
    for p in sorted(folder.glob("*.tsv")):
        try:
            h=pd.read_csv(p,sep="\t",nrows=2)
            if set(required).issubset(h.columns): return pd.read_csv(p,sep="\t",low_memory=False)
        except Exception: pass
    raise FileNotFoundError(f"No table in {folder} has {required}")
def find_json(folder,contains=None):
    for p in sorted(Path(folder).glob("*.json")):
        d=json.loads(p.read_text())
        if contains is None or contains in d: return d
    raise FileNotFoundError(folder)
def output_dir(root):
    p=Path(root)/"figures"/"main_manuscript_figures"; p.mkdir(parents=True,exist_ok=True); return p
def save(fig,out,stem):
    fig.savefig(out/f"{stem}.pdf",bbox_inches="tight")
    fig.savefig(out/f"{stem}.svg",bbox_inches="tight")
    fig.savefig(out/f"{stem}.png",dpi=600,bbox_inches="tight")
    fig.savefig(out/f"{stem}.tiff",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
def save_panels(fig,axes,out,names):
    """Export axes as independent publication panels; no composite is written."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for ax,name in zip(axes,names):
        box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.10,1.13)
        for ext in ["pdf","svg"]: fig.savefig(out/f"{name}.{ext}",bbox_inches=box)
        fig.savefig(out/f"{name}.png",dpi=600,bbox_inches=box)
        fig.savefig(out/f"{name}.tiff",dpi=600,bbox_inches=box,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
