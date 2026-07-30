"""Settings page — system information and runtime diagnostics."""

from __future__ import annotations
import subprocess
import psutil
import streamlit as st
from alphafold_fusion import __version__


def render() -> None:
    st.markdown('<div class="sub-header">⚙️ System Info</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("CPU Cores", psutil.cpu_count())
        st.metric("RAM", f"{psutil.virtual_memory().total / 2**30:.1f} GB")
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=False)
            gpu_name = "None"
            if r.returncode == 0 and r.stdout.strip():
                gpu_name = r.stdout.strip().split("\n")[0]
            st.metric("GPU", gpu_name)
        except Exception:
            st.metric("GPU", "None")
    with c2:
        st.metric("Version", __version__)
        try:
            import jax
            st.caption(f"JAX: {jax.default_backend()} | {jax.devices()}")
        except Exception:
            st.caption("JAX: not available")