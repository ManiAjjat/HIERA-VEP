import os
#!/usr/bin/env python3
"""Step 57F: curated redesign of final main and supplementary multipanel figures."""
from pathlib import Path
import argparse, base64, io, json, re, shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont


MAIN_SPECS = [
    ("Figure_1", "Study design and data architecture", [
        ("a", ["framework"], "Study and modeling framework"),
        ("b", ["dataset", "manifest"], "Dataset and assay composition"),
        ("c", ["split", "leakage"], "Leakage-safe evaluation architecture"),
        ("d", ["coordinate", "resolvability"], "GRCh38 coordinate eligibility"),
        ("e", ["encode", "promoter"], "ENCODE cCRE and promoter context"),
    ]),
    ("Figure_2", "Locked predictive evaluation and comparative performance", [
        ("a", ["locked", "context", "point"], "Locked context-aware performance"),
        ("b", ["context", "sequence", "paired"], "Paired context gain"),
        ("c", ["genomic", "language", "comparison"], "Frozen genomic-language-model comparison"),
        ("d", ["nested", "grouped", "selection"], "Nested grouped selection"),
        ("e", ["feature", "family", "ablation"], "Locked feature-family ablation"),
        ("f", ["calibration"], "Locked calibration"),
    ]),
    ("Figure_3", "External transfer, adaptation and condition dependence", [
        ("a", ["leave", "study", "external"], "Locked zero-shot external evaluation"),
        ("b", ["external", "paired"], "Paired external comparison"),
        ("c", ["assay", "adaptive", "point"], "Assay-adaptive modeling"),
        ("d", ["few", "shot", "learning"], "Few-shot learning curve"),
        ("e", ["condition", "agreement"], "Paired SUL1 condition agreement"),
        ("f", ["condition", "differential"], "Condition-differential prediction"),
    ]),
    ("Figure_4", "Exact model interpretation and feature dependence", [
        ("a", ["block", "contribution"], "Exact additive block contributions"),
        ("b", ["block", "removal"], "Block-removal sensitivity"),
        ("c", ["attribution"], "Variant-level attribution structure"),
        ("d", ["coefficient"], "Strongest interpretable model coefficients"),
        ("e", ["counterfactual"], "Context counterfactual sensitivity"),
    ]),
    ("Figure_5", "Biological concordance and model limitations", [
        ("a", ["ccre", "annotation"], "Regulatory-class annotation landscape"),
        ("b", ["conservation"], "Phylogenetic conservation landscape"),
        ("c", ["position", "motif", "conservation"], "Position-level motif and conservation concordance"),
        ("d", ["fungal", "motif"], "Condition-dependent fungal motif context"),
        ("e", ["gtex", "eqtl"], "Exact GTEx overlap"),
        ("f", ["uncertainty", "failure"], "Uncertainty and failure modes"),
    ]),
]

SUPPLEMENT_SPECS = [
    ("Supplementary_Figure_S1", "Dataset quality control and split integrity",
     ["audit", "manifest", "registry", "split", "qc"]),
    ("Supplementary_Figure_S2", "Model tuning and robustness",
     ["tuning", "outer", "bootstrap", "robustness", "kmer"]),
    ("Supplementary_Figure_S3", "External dataset inventory and harmonization",
     ["external", "inventory", "harmonization", "score_set", "dataset"]),
    ("Supplementary_Figure_S4", "External context sensitivity and adaptation",
     ["padding", "context", "adapt", "transfer", "few_shot"]),
    ("Supplementary_Figure_S5", "Motif threshold and enrichment robustness",
     ["motif", "threshold", "enrichment", "maxT", "fungal"]),
    ("Supplementary_Figure_S6", "Genomic annotation and conservation",
     ["encode", "ccre", "promoter", "conservation", "coordinate"]),
    ("Supplementary_Figure_S7", "Attribution, ablation and failure diagnostics",
     ["attribution", "ablation", "calibration", "uncertainty", "failure", "residual"]),
    ("Supplementary_Figure_S8", "Reproducibility and release audit",
     ["release", "reproducibility", "coverage", "checksum", "deployment"]),
]

EXCLUDE = ["contact_sheet", "quality_audit", "final_coverage_atlas", "visualization_routing",
           "module_output_atlas", "inventory_disposition"]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--supplement-panels", type=int, default=6)
    return p.parse_args()


def crop(image, pad=28):
    image = image.convert("RGB")
    diff = ImageChops.difference(image, Image.new("RGB", image.size, "white")).convert("L")
    diff = diff.point(lambda x: 255 if x > 9 else 0)
    box = diff.getbbox()
    if box is None:
        return image
    return image.crop((max(0, box[0]-pad), max(0, box[1]-pad),
                       min(image.width, box[2]+pad), min(image.height, box[3]+pad)))


def quality(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
            thumb = image.convert("L"); thumb.thumbnail((500, 500))
            a = np.asarray(thumb, dtype=float)
            ink, contrast = float((a < 245).mean()), float(a.std())
            return min(width/1600, 1.5) + min(height/1100, 1.5) + min(ink*20, 1.5) + min(contrast/35, 1.5)
    except Exception:
        return -999


def discover(root):
    records = []
    for path in sorted((root / "figures").rglob("*.png")):
        lower = str(path).lower()
        if "final-manuscript" in lower or "quality_control" in lower or any(x in lower for x in EXCLUDE):
            continue
        score = quality(path)
        if score < 2:
            continue
        records.append({"path": path, "relative": str(path.relative_to(root)),
                        "text": (path.stem + " " + str(path.parent)).lower(), "quality": score,
                        "origin": "main" if "main" in lower and "supplement" not in lower else "supplementary"})
    data = pd.DataFrame(records)
    if not len(data):
        raise RuntimeError("No eligible figure panels found")
    return data


def match_score(text, tokens):
    normalized = text.replace("_", " ").replace("-", " ")
    score = 0
    for token in tokens:
        token = token.replace("_", " ")
        if token in normalized:
            score += 5 if " " in token else 2
    return score


def select_main(candidates):
    used, selections = set(), []
    for figure, title, panels in MAIN_SPECS:
        for label, tokens, rationale in panels:
            ranked = candidates.copy()
            ranked["match"] = ranked.text.apply(lambda x: match_score(x, tokens))
            ranked["total"] = ranked["match"] + ranked.quality*.2 + (ranked.origin == "main")*.5
            ranked = ranked.sort_values(["total", "quality"], ascending=False)
            available = ranked[~ranked.relative.isin(used)]
            if not len(available):
                raise RuntimeError(f"No unused panel available for {figure}{label}")
            best = available.iloc[0]
            if best.match == 0:
                raise RuntimeError(f"No scientifically matching panel found for {figure}{label}: {rationale}")
            used.add(best.relative)
            selections.append({"figure": figure, "figure_title": title, "panel": label,
                               "source_path": best.relative, "rationale": rationale,
                               "semantic_match_score": int(best.match), "quality_score": float(best.quality),
                               "promoted_from_supplement": bool(best.origin == "supplementary")})
    return pd.DataFrame(selections), used


def select_supplement(candidates, main_used, panel_count):
    used = set(main_used)
    rows = []
    for figure, title, tokens in SUPPLEMENT_SPECS:
        ranked = candidates.copy()
        ranked["match"] = ranked.text.apply(lambda x: match_score(x, tokens))
        ranked["total"] = ranked["match"] + ranked.quality*.2
        ranked = ranked[(ranked["match"] > 0) & (~ranked.relative.isin(used))]
        ranked = ranked.sort_values(["total", "quality"], ascending=False).head(panel_count)
        if len(ranked) < 4:
            continue
        for i, (_, best) in enumerate(ranked.iterrows()):
            used.add(best.relative)
            rows.append({"figure": figure, "figure_title": title, "panel": chr(97+i),
                         "source_path": best.relative, "rationale": title,
                         "semantic_match_score": int(best.match), "quality_score": float(best.quality)})
    return pd.DataFrame(rows)


def load_font(size):
    for path in [Path("fonts/Arial.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def panel_tile(root, row, width, height):
    with Image.open(root / row.source_path) as source:
        image = crop(source)
        image.thumbnail((width-60, height-60), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (width, height), "white")
        tile.paste(image, ((width-image.width)//2, (height-image.height)//2))
    draw = ImageDraw.Draw(tile)
    fnt = load_font(max(48, int(height*.052)))
    label = str(row.panel)
    box = draw.textbbox((0, 0), label, font=fnt)
    draw.rounded_rectangle((14, 10, box[2]+42, box[3]+34), radius=10, fill="white", outline="#333333", width=2)
    draw.text((26, 16), label, font=fnt, fill="black")
    return tile


def layout_rectangles(n):
    if n == 4:
        return [(0,0,1,1),(1,0,1,1),(0,1,1,1),(1,1,1,1)], 2, 2
    if n == 5:
        return [(0,0,2,1),(2,0,1,1),(0,1,1,1),(1,1,1,1),(2,1,1,1)], 3, 2
    if n == 6:
        return [(0,0,1,1),(1,0,1,1),(2,0,1,1),(0,1,1,1),(1,1,1,1),(2,1,1,1)], 3, 2
    rectangles=[]
    for i in range(n):rectangles.append((i%3,i//3,1,1))
    return rectangles,3,int(np.ceil(n/3))


def compose(root, rows):
    n = len(rows); rects, cols, lines = layout_rectangles(n)
    unit_w, unit_h, gutter = 1800, 1320, 16
    canvas = Image.new("RGB", (cols*unit_w+(cols-1)*gutter,
                               lines*unit_h+(lines-1)*gutter), "white")
    for (_, row), (x,y,w,h) in zip(rows.iterrows(), rects):
        width, height = w*unit_w+(w-1)*gutter, h*unit_h+(h-1)*gutter
        tile = panel_tile(root, row, width, height)
        canvas.paste(tile, (x*(unit_w+gutter), y*(unit_h+gutter)))
    return canvas


def svg(image, path):
    stream=io.BytesIO(); image.save(stream,"PNG",dpi=(600,600),optimize=True)
    data=base64.b64encode(stream.getvalue()).decode("ascii")
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {image.width} {image.height}"><image width="{image.width}" height="{image.height}" href="data:image/png;base64,{data}"/></svg>',encoding="utf-8")


def export(image, folder, stem):
    image.save(folder/f"{stem}.png",dpi=(600,600),optimize=True)
    image.save(folder/f"{stem}.tiff",dpi=(600,600),compression="tiff_lzw")
    image.save(folder/f"{stem}.pdf",resolution=600)
    svg(image,folder/f"{stem}.svg")


def copy_tables(root, package):
    source=root/"tables/final_manuscript_tables"
    pairs=[("HIERA_VEP_Main_Manuscript_Tables.docx","final-ms tables.docx"),
           ("HIERA_VEP_Supplementary_Tables.docx","supplementary tables.docx")]
    status=[]
    for src,dst in pairs:
        if (source/src).exists():shutil.copy2(source/src,package/dst);state="COPIED"
        else:state="MISSING"
        status.append({"file":dst,"status":state,"source":str(source/src)})
    return status


def caption_file(main, supplement):
    lines=["# Curated final figure captions",""]
    for collection,label in [(main,"Main"),(supplement,"Supplementary")]:
        if not len(collection):continue
        for fig,g in collection.groupby("figure",sort=True):
            lines += [f"## {fig.replace('_',' ')}. {g.figure_title.iloc[0]}",""]
            for _,r in g.sort_values("panel").iterrows():
                lines.append(f"**({r.panel})** {r.rationale}. Source: `{r.source_path}`.")
            lines += ["","Final wording must be checked against the locked results registry.",""]
    return "\n".join(lines)


def main():
    args=arguments();root=args.project.resolve();package=root/"final-manuscript"
    main_dir=package/"ms figures";supp_dir=package/"supplementary figures"
    main_dir.mkdir(parents=True,exist_ok=True);supp_dir.mkdir(parents=True,exist_ok=True)
    candidates=discover(root);main,used=select_main(candidates)
    supplement=select_supplement(candidates,used,args.supplement_panels)
    exports=[]
    for collection,folder,kind in [(main,main_dir,"main"),(supplement,supp_dir,"supplementary")]:
        for figure,group in collection.groupby("figure",sort=True):
            group=group.sort_values("panel");image=compose(root,group)
            stem=f"{figure}_curated_final";export(image,folder,stem)
            exports.append({"figure":figure,"kind":kind,"panels":len(group),"stem":stem,
                            "dimensions_px":f"{image.width}x{image.height}"})
            print(f"{figure}: {len(group)} panels | {image.width}x{image.height}px")
    main.to_csv(package/"curated_main_figure_panel_manifest.tsv",sep="\t",index=False)
    supplement.to_csv(package/"curated_supplementary_multipanel_manifest.tsv",sep="\t",index=False)
    pd.DataFrame(exports).to_csv(package/"curated_figure_export_manifest.tsv",sep="\t",index=False)
    table_status=copy_tables(root,package)
    pd.DataFrame(table_status).to_csv(package/"final_table_copy_status.tsv",sep="\t",index=False)
    (package/"CURATED_FIGURE_CAPTIONS.md").write_text(caption_file(main,supplement),encoding="utf-8")
    summary={"main_figures":int(main.figure.nunique()),"main_panels":int(len(main)),
             "supplementary_multipanel_figures":int(supplement.figure.nunique()) if len(supplement) else 0,
             "supplementary_panels":int(len(supplement)),"main_tables":table_status[0]["status"],
             "supplementary_tables":table_status[1]["status"],"package":str(package),
             "selection_policy":"Curated by scientific narrative, evidentiary role, visual quality and nonduplication; null and limitation panels retained.",
             "required_review":"Inspect composites at 100% zoom and verify every caption against the locked registry."}
    (package/"57F_curated_redesign_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nSTEP 57F CURATED FINAL FIGURE REDESIGN");print(json.dumps(summary,indent=2));print("Saved:",package)


if __name__=="__main__":main()
