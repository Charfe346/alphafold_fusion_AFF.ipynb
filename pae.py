"""PAE (Predicted Aligned Error) heatmap generation and pickle loading."""

from __future__ import annotations
import json, pickle, re
from pathlib import Path
from typing import Optional, Union
import numpy as np
import plotly.express as px


def pae_heatmap(source: Union[str, Path, np.ndarray],
                title: str = "PAE Heatmap"):
    try:
        if isinstance(source, (str, Path)):
            with open(source) as f:
                j = json.load(f)
            mat = j.get("pae") or j.get("predicted_aligned_error")
        else:
            mat = source
        if mat is None:
            return None
        arr = np.array(mat)
        fig = px.imshow(
            arr, color_continuous_scale="Viridis", origin="lower",
            labels={"color": "PAE (Å)"}, title=title,
            zmin=0, zmax=max(30, float(np.nanmax(arr))),
            aspect="auto",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig
    except Exception:
        return None


def pae_from_pkl(job_dir: Path,
                 model_id: str) -> Optional[np.ndarray]:
    if not job_dir.exists() or not model_id:
        return None
    # model_id may be stored as "model_1" or "model1"; ColabFold pkl
    # filenames use "result_model_1_...". Normalise to the underscore
    # form so the glob matches regardless of the stored variant.
    mid_glob = re.sub(r"model_?(\d+)", r"model_\1", model_id)
    files = (list(job_dir.rglob(f"result_{mid_glob}*.pkl"))
             or list(job_dir.rglob(f"result_{model_id}*.pkl")))
    if not files:
        return None
    try:
        with open(
            sorted(files, key=lambda p: p.stat().st_mtime)[-1], "rb"
        ) as f:
            # SECURITY: pickle.load executes arbitrary code. Only load
            # .pkl files produced by our own ColabFold runs, never from
            # untrusted sources.
            d = pickle.load(f)
        pae = d.get("predicted_aligned_error") or d.get("pae")
        if pae is None and "pae_output" in d:
            pae = (d["pae_output"] or {}).get("pae")
        return np.array(pae) if pae is not None else None
    except Exception:
        return None