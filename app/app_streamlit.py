# Streamlit App — Scientific Computing for All
# Milestone 2 (masks) + Implicit Diffusion + fixes
# ------------------------------------------------------------
# Run:  streamlit run app_streamlit.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import time
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image

# --- import the engine (Stage 1 core, patched for masks/linearity) ---
try:
    from engine.stage1_core import (
        Grid2D, Boundary,
        Poisson2D, Diffusion2D, Wave2D,
        arr_like, _is_homog_dirichlet
    )
except Exception:
    from stage1_core import (
        Grid2D, Boundary,
        Poisson2D, Diffusion2D, Wave2D,
        arr_like, _is_homog_dirichlet
    )

# ---- Streamlit rerun helper (version-safe) ----
def _rerun():
    try:
        st.rerun()               # Streamlit ≥ 1.30
    except AttributeError:
        st.experimental_rerun()  # older versions

# Optional: show backend hint (engine auto-handles fallback)
GPU_AVAILABLE = (os.environ.get("CPU_FALLBACK", "0") != "1")

# ----------------- Utility helpers -----------------
def to_numpy(a):
    """Works for numpy or cupy arrays."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)

def export_npz(field, meta: dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, field=to_numpy(field), meta=meta)
    buf.seek(0)
    return buf

def export_png(field, title="field", cmap="viridis"):
    import matplotlib.pyplot as plt
    arr = to_numpy(field)
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf

# ----------------- Mask helpers -----------------
def blank_mask(Ny, Nx):
    return np.ones((Ny, Nx), dtype=bool)

def add_rect(mask, x0, x1, y0, y1, inside=True):
    Ny, Nx = mask.shape
    ix0 = max(0, min(Nx-1, int(x0)))
    ix1 = max(0, min(Nx-1, int(x1)))
    iy0 = max(0, min(Ny-1, int(y0)))
    iy1 = max(0, min(Ny-1, int(y1)))
    if ix0 > ix1: ix0, ix1 = ix1, ix0
    if iy0 > iy1: iy0, iy1 = iy1, iy0
    if inside:
        mask[:, :] = False
        mask[iy0:iy1+1, ix0:ix1+1] = True
    else:
        mask[iy0:iy1+1, ix0:ix1+1] = False
    return mask

def add_circle(mask, cx, cy, r, inside=True):
    Ny, Nx = mask.shape
    Y, X = np.ogrid[:Ny, :Nx]
    sel = (X - cx)**2 + (Y - cy)**2 <= r**2
    if inside:
        mask[:, :] = False
        mask[sel] = True
    else:
        mask[sel] = False
    return mask

def load_png_mask(file, Ny, Nx):
    img = Image.open(file).convert("L").resize((Nx, Ny))
    arr = np.array(img)
    return (arr > 127)

def overlay_mask_preview(img, mask):
    """Dim/transparent outside mask for nicer visualization."""
    if mask is None:
        return img
    out = to_numpy(img).copy()
    out[~mask] = np.nan  # Plotly shows gaps for NaNs
    return out

# ----------------- Session state init -----------------
def ensure_state():
    ss = st.session_state
    # core
    ss.setdefault("pde", "Poisson")
    ss.setdefault("Nx", 256)
    ss.setdefault("Ny", 256)
    ss.setdefault("dtype", "float32")
    # BCs
    ss.setdefault("bc_left", ("Dirichlet", 0.0))
    ss.setdefault("bc_right", ("Dirichlet", 0.0))
    ss.setdefault("bc_bottom", ("Dirichlet", 0.0))
    ss.setdefault("bc_top", ("Dirichlet", 0.0))
    # Poisson
    ss.setdefault("poisson_tol", 1e-6)
    ss.setdefault("poisson_maxiter", 2000)
    # Diffusion
    ss.setdefault("diff_alpha", 1.0)
    ss.setdefault("diff_cfl", 0.9)
    ss.setdefault("diff_steps_per_frame", 5)
    ss.setdefault("diff_use_implicit", False)
    ss.setdefault("diff_dt_implicit", 1e-2)
    # Wave
    ss.setdefault("wave_c", 1.0)
    ss.setdefault("wave_cfl", 0.9)
    ss.setdefault("wave_steps_per_frame", 5)
    # Display
    ss.setdefault("autoscale", True)
    ss.setdefault("vmin", 0.0)
    ss.setdefault("vmax", 1.0)
    # Runtime
    ss.setdefault("field", None)          # current field (u) — NumPy for UI
    ss.setdefault("field_prev", None)     # for wave (u_{n-1})
    ss.setdefault("engine", None)         # active solver object
    ss.setdefault("grid", None)
    ss.setdefault("boundary", None)
    ss.setdefault("running", False)       # run/pause flag
    ss.setdefault("timestep", 0)
    ss.setdefault("dt_info", "")
    ss.setdefault("last_runtime", None)
    ss.setdefault("last_solve_info", None)
    # Mask
    ss.setdefault("use_mask", False)
    ss.setdefault("phi_mask", 0.0)
    ss.setdefault("mask_arr", None)

ensure_state()

# ----------------- Sidebar: Presets -----------------
st.sidebar.title("Presets")
preset = st.sidebar.radio("Configuration preset", ["Beginner", "Balanced", "Accurate"], index=1)
if preset == "Beginner":
    default_N = 128; default_dtype = "float32"; default_tol = 1e-4
    default_cfl = 0.8; default_steps = 5
elif preset == "Accurate":
    default_N = 384; default_dtype = "float64"; default_tol = 1e-8
    default_cfl = 0.95; default_steps = 10
else:  # Balanced
    default_N = 256; default_dtype = "float32"; default_tol = 1e-6
    default_cfl = 0.9; default_steps = 5

if st.sidebar.button("Apply preset"):
    st.session_state.Nx = st.session_state.Ny = default_N
    st.session_state.dtype = default_dtype
    st.session_state.poisson_tol = default_tol
    st.session_state.diff_cfl = default_cfl
    st.session_state.wave_cfl = default_cfl
    st.session_state.diff_steps_per_frame = default_steps
    st.session_state.wave_steps_per_frame = default_steps

# ----------------- Sidebar: Model -----------------
st.sidebar.title("Model")
pde = st.sidebar.selectbox("PDE", ["Poisson", "Diffusion", "Wave"],
                           index=["Poisson","Diffusion","Wave"].index(st.session_state.pde))
st.session_state.pde = pde

colN1, colN2 = st.sidebar.columns(2)
Nx = colN1.number_input("Nx", min_value=32, max_value=2048, value=st.session_state.Nx, step=32)
Ny = colN2.number_input("Ny", min_value=32, max_value=2048, value=st.session_state.Ny, step=32)
dtype = st.sidebar.selectbox("Precision", ["float32", "float64"],
                             index=0 if st.session_state.dtype == "float32" else 1)
st.session_state.Nx, st.session_state.Ny, st.session_state.dtype = Nx, Ny, dtype

# ----------------- Sidebar: Boundary Conditions -----------------
st.sidebar.title("Boundary Conditions")
def bc_block(label, current):
    kind = st.sidebar.selectbox(f"{label} type", ["Dirichlet", "Neumann", "Robin"],
                                key=f"{label}_kind", index=["Dirichlet","Neumann","Robin"].index(current[0]))
    if kind == "Dirichlet":
        val = st.sidebar.number_input(f"{label} φ", value=float(current[1] if isinstance(current[1], (float,int)) else 0.0))
        return (kind, float(val))
    elif kind == "Neumann":
        val = st.sidebar.number_input(f"{label} q (∂u/∂n)", value=float(current[1] if isinstance(current[1], (float,int)) else 0.0))
        return (kind, float(val))
    else:
        a = st.sidebar.number_input(f"{label} a", value=float(current[1][0] if isinstance(current[1], tuple) else 1.0))
        b = st.sidebar.number_input(f"{label} b (≠0)", value=float(current[1][1] if isinstance(current[1], tuple) else 1.0))
        c = st.sidebar.number_input(f"{label} c", value=float(current[1][2] if isinstance(current[1], tuple) else 0.0))
        return (kind, (float(a), float(b), float(c)))

st.session_state.bc_left   = bc_block("Left",   st.session_state.bc_left)
st.session_state.bc_right  = bc_block("Right",  st.session_state.bc_right)
st.session_state.bc_bottom = bc_block("Bottom", st.session_state.bc_bottom)
st.session_state.bc_top    = bc_block("Top",    st.session_state.bc_top)

# ----------------- Sidebar: Parameters -----------------
st.sidebar.title("Parameters")
if pde == "Poisson":
    tol = st.sidebar.number_input("CG tolerance", value=float(st.session_state.poisson_tol), format="%.1e")
    maxiter = st.sidebar.number_input("CG maxiter", min_value=1, max_value=20000, value=st.session_state.poisson_maxiter, step=100)
    st.session_state.poisson_tol = tol
    st.session_state.poisson_maxiter = maxiter
elif pde == "Diffusion":
    alpha = st.sidebar.number_input("Diffusivity α", value=float(st.session_state.diff_alpha), min_value=0.0, step=0.1)
    cfl = st.sidebar.slider("CFL fraction (≤1.0)", min_value=0.1, max_value=1.0, value=float(st.session_state.diff_cfl), step=0.05)
    steps_per_frame = st.sidebar.number_input("Steps per frame", value=st.session_state.diff_steps_per_frame, min_value=1, max_value=200)
    # Implicit option
    use_implicit = st.sidebar.checkbox("Use implicit (Backward Euler)", value=st.session_state.diff_use_implicit)
    dt_imp = st.sidebar.number_input("dt (implicit)", value=float(st.session_state.diff_dt_implicit), min_value=1e-6, format="%.3e")
    st.session_state.diff_alpha = alpha
    st.session_state.diff_cfl = cfl
    st.session_state.diff_steps_per_frame = steps_per_frame
    st.session_state.diff_use_implicit = use_implicit
    st.session_state.diff_dt_implicit = dt_imp
else:
    c_speed = st.sidebar.number_input("Wave speed c", value=float(st.session_state.wave_c), min_value=0.0, step=0.1)
    cfl = st.sidebar.slider("CFL fraction (≤1.0)", min_value=0.1, max_value=1.0, value=float(st.session_state.wave_cfl), step=0.05)
    steps_per_frame = st.sidebar.number_input("Steps per frame", value=st.session_state.wave_steps_per_frame, min_value=1, max_value=200)
    st.session_state.wave_c = c_speed
    st.session_state.wave_cfl = cfl
    st.session_state.wave_steps_per_frame = steps_per_frame

# ----------------- Sidebar: Geometry Mask -----------------
st.sidebar.title("Geometry Mask")
st.session_state.use_mask = st.sidebar.checkbox("Use mask (Dirichlet outside)", value=st.session_state.use_mask)
st.session_state.phi_mask = st.sidebar.number_input("φ outside mask", value=float(st.session_state.phi_mask))

if st.session_state.use_mask:
    Ny_i = int(st.session_state.Ny); Nx_i = int(st.session_state.Nx)
    if st.session_state.mask_arr is None or st.session_state.mask_arr.shape != (Ny_i, Nx_i):
        st.session_state.mask_arr = blank_mask(Ny_i, Nx_i)

    st.sidebar.subheader("Draw shapes")
    with st.sidebar.expander("Rectangle (keep only or cut out)"):
        x0 = st.number_input("x0 (cols)", min_value=0, max_value=Nx_i-1, value=0, key="r_x0")
        x1 = st.number_input("x1 (cols)", min_value=0, max_value=Nx_i-1, value=Nx_i-1, key="r_x1")
        y0 = st.number_input("y0 (rows)", min_value=0, max_value=Ny_i-1, value=0, key="r_y0")
        y1 = st.number_input("y1 (rows)", min_value=0, max_value=Ny_i-1, value=Ny_i-1, key="r_y1")
        col_keep, col_cut = st.columns(2)
        if col_keep.button("Keep only rectangle"):
            st.session_state.mask_arr = add_rect(blank_mask(Ny_i, Nx_i), x0, x1, y0, y1, inside=True)
        if col_cut.button("Cut out rectangle"):
            st.session_state.mask_arr = add_rect(st.session_state.mask_arr.copy(), x0, x1, y0, y1, inside=False)

    with st.sidebar.expander("Circle (keep only or cut out)"):
        cx = st.number_input("cx (cols)", min_value=0, max_value=Nx_i-1, value=Nx_i//2, key="c_cx")
        cy = st.number_input("cy (rows)", min_value=0, max_value=Ny_i-1, value=Ny_i//2, key="c_cy")
        r  = st.number_input("radius (px)", min_value=1, max_value=min(Nx_i,Ny_i)//2, value=min(Nx_i,Ny_i)//4, key="c_r")
        col_keep2, col_cut2 = st.columns(2)
        if col_keep2.button("Keep only circle"):
            st.session_state.mask_arr = add_circle(blank_mask(Ny_i, Nx_i), cx, cy, r, inside=True)
        if col_cut2.button("Cut out circle"):
            st.session_state.mask_arr = add_circle(st.session_state.mask_arr.copy(), cx, cy, r, inside=False)

    st.sidebar.subheader("Upload mask (PNG, white=in)")
    up = st.sidebar.file_uploader("Binary/gray PNG", type=["png"])
    if up is not None:
        st.session_state.mask_arr = load_png_mask(up, Ny_i, Nx_i)

    if st.sidebar.button("Clear mask"):
        st.session_state.mask_arr = blank_mask(Ny_i, Nx_i)

# ----------------- Header -----------------
st.title("Scientific Computing for All: A Simple PDE Solver")
st.caption(f"Backend: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy)'} • PDE: {pde} • Grid: {Nx}×{Ny} • dtype: {dtype}")

# ----------------- Build grid/BC/engine -----------------
def build_solvers():
    g = Grid2D(Nx=int(Nx), Ny=int(Ny), Lx=1.0, Ly=1.0)
    bc = Boundary(
        left=st.session_state.bc_left,
        right=st.session_state.bc_right,
        bottom=st.session_state.bc_bottom,
        top=st.session_state.bc_top
    )
    mask = st.session_state.mask_arr if st.session_state.use_mask else None
    phi_m = float(st.session_state.phi_mask)
    st.session_state.grid = g
    st.session_state.boundary = bc
    if pde == "Poisson":
        st.session_state.engine = Poisson2D(g, bc, dtype=dtype, mask=mask, phi_mask=phi_m)
    elif pde == "Diffusion":
        st.session_state.engine = Diffusion2D(g, bc, alpha=st.session_state.diff_alpha, dtype=dtype, mask=mask, phi_mask=phi_m)
    else:
        st.session_state.engine = Wave2D(g, bc, c=st.session_state.wave_c, dtype=dtype, mask=mask, phi_mask=phi_m)

def reset_field():
    g = st.session_state.grid
    if pde == "Poisson":
        st.session_state.field = None
        st.session_state.field_prev = None
        st.session_state.timestep = 0
    elif pde == "Diffusion":
        u = arr_like(g, dtype=np.float32 if dtype=="float32" else np.float64)
        r = max(2, min(g.Nx, g.Ny)//50)
        u[g.Ny//2-r:g.Ny//2+r, g.Nx//2-r:g.Nx//2+r] = 1.0
        st.session_state.field = to_numpy(u)
        st.session_state.field_prev = None
        st.session_state.timestep = 0
    else:
        # Wave: Gaussian bump (NumPy for UI; engine will convert internally)
        h = g.hx
        y = np.arange(1, g.Ny+1) * h
        x = np.arange(1, g.Nx+1) * h
        X, Y = np.meshgrid(x, y, indexing="xy")
        sigma = 0.10
        u0 = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(2*sigma**2))
        st.session_state.field = u0.astype(np.float32 if dtype=="float32" else np.float64)
        st.session_state.field_prev = None
        st.session_state.timestep = 0

# Rebuild on Apply or first load
if st.button("Apply configuration"):
    build_solvers()
    reset_field()
if st.session_state.engine is None:
    build_solvers()
    reset_field()

g = st.session_state.grid
engine = st.session_state.engine

# ----------------- Controls -----------------
colA, colB, colC, colD = st.columns(4)

if pde == "Poisson":
    if colA.button("Solve"):
        f = engine.rhs()  # analytic RHS
        t0 = time.perf_counter()
        u, info = engine.solve_cg(f, tol=st.session_state.poisson_tol, maxiter=st.session_state.poisson_maxiter)
        t1 = time.perf_counter()
        st.session_state.field = to_numpy(u)
        st.session_state.last_runtime = t1 - t0
        st.session_state.last_solve_info = int(info)
        st.session_state.timestep = 0

elif pde == "Diffusion":
    run_clicked = colA.button("Run/Pause")
    step_clicked = colB.button("Step")
    reset_clicked = colC.button("Reset")
    if reset_clicked:
        reset_field()
    if run_clicked:
        st.session_state.running = not st.session_state.running
    if step_clicked:
        st.session_state.running = False
        if st.session_state.diff_use_implicit:
            dt = float(st.session_state.diff_dt_implicit)
            for _ in range(st.session_state.diff_steps_per_frame):
                st.session_state.field = to_numpy(
                    engine.step_implicit_be(st.session_state.field, dt)
                )
                st.session_state.timestep += 1
            st.session_state.dt_info = f"implicit dt={dt:.3e}"
        else:
            dt = st.session_state.diff_cfl * engine.dt_stable
            for _ in range(st.session_state.diff_steps_per_frame):
                st.session_state.field = to_numpy(
                    engine.step_explicit(st.session_state.field, dt)
                )
                st.session_state.timestep += 1
            st.session_state.dt_info = f"explicit dt={dt:.3e}"

else:  # Wave
    run_clicked = colA.button("Run/Pause")
    step_clicked = colB.button("Step")
    reset_clicked = colC.button("Reset")
    if reset_clicked:
        reset_field()
    if run_clicked:
        st.session_state.running = not st.session_state.running
    if step_clicked:
        st.session_state.running = False
        dt = st.session_state.wave_cfl * engine.dt_cfl
        if st.session_state.field_prev is None:
            # engine.lap returns backend array (CuPy on GPU); convert to NumPy for UI math
            lap_u0 = to_numpy(engine.lap(st.session_state.field))
            st.session_state.field_prev = st.session_state.field + 0.5*(engine.c**2)*(dt**2)*lap_u0
        u_next = to_numpy(engine.step_leapfrog(st.session_state.field, st.session_state.field_prev, dt))
        st.session_state.field_prev, st.session_state.field = st.session_state.field, u_next
        st.session_state.timestep += 1
        st.session_state.dt_info = f"dt={dt:.3e}"

# Continuous running (Diffusion/Wave)
if st.session_state.running and pde in ("Diffusion", "Wave"):
    if pde == "Diffusion":
        if st.session_state.diff_use_implicit:
            dt = float(st.session_state.diff_dt_implicit)
            steps = st.session_state.diff_steps_per_frame
            for _ in range(steps):
                st.session_state.field = to_numpy(
                    engine.step_implicit_be(st.session_state.field, dt)
                )
                st.session_state.timestep += 1
            st.session_state.dt_info = f"implicit dt={dt:.3e}"
        else:
            dt = st.session_state.diff_cfl * engine.dt_stable
            steps = st.session_state.diff_steps_per_frame
            for _ in range(steps):
                st.session_state.field = to_numpy(
                    engine.step_explicit(st.session_state.field, dt)
                )
                st.session_state.timestep += 1
            st.session_state.dt_info = f"explicit dt={dt:.3e}"
    else:  # Wave
        dt = st.session_state.wave_cfl * engine.dt_cfl
        steps = st.session_state.wave_steps_per_frame
        for _ in range(steps):
            if st.session_state.field_prev is None:
                lap_u0 = to_numpy(engine.lap(st.session_state.field))
                st.session_state.field_prev = st.session_state.field + 0.5*(engine.c**2)*(dt**2)*lap_u0
            u_next = to_numpy(engine.step_leapfrog(st.session_state.field, st.session_state.field_prev, dt))
            st.session_state.field_prev, st.session_state.field = st.session_state.field, u_next
            st.session_state.timestep += 1
        st.session_state.dt_info = f"dt={dt:.3e}"
    _rerun()

# ----------------- Display controls -----------------
st.subheader("Display")
c_aut, c_min, c_max = st.columns([1,1,1])
st.session_state.autoscale = c_aut.checkbox("Autoscale", value=st.session_state.autoscale)
if not st.session_state.autoscale:
    st.session_state.vmin = c_min.number_input("vmin", value=float(st.session_state.vmin))
    st.session_state.vmax = c_max.number_input("vmax", value=float(st.session_state.vmax))

# ----------------- Main plot -----------------
st.subheader("Field")
arr = st.session_state.field
mask_preview = st.session_state.mask_arr if st.session_state.use_mask else None
if arr is None and pde == "Poisson":
    st.info("Click **Solve** to compute the steady-state field.")
else:
    img = to_numpy(arr if arr is not None else np.zeros((g.Ny, g.Nx)))
    vargs = {}
    if not st.session_state.autoscale:
        vargs = dict(zmin=st.session_state.vmin, zmax=st.session_state.vmax)
    fig = px.imshow(overlay_mask_preview(img, mask_preview),
                    origin="lower", aspect="auto", color_continuous_scale="Viridis", **vargs)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_colorbar=dict(title="u"))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Status / metrics -----------------
st.markdown("### Status")
cols = st.columns(4)
cols[0].metric("Timesteps", f"{st.session_state.timestep}")
cols[1].metric("Running", "Yes" if st.session_state.running else "No")
cols[2].metric("dt", st.session_state.dt_info if st.session_state.dt_info else "-")
if pde == "Poisson":
    rt = st.session_state.last_runtime
    info = st.session_state.last_solve_info
    cols[3].metric("Solve time", f"{rt:.3f} s" if rt else "-")
    st.caption(f"CG info: {info} (0=converged)" if info is not None else "CG info: -")
else:
    cols[3].metric("Backend", "GPU" if GPU_AVAILABLE else "CPU")

# ----------------- Config (Save / Load) -----------------
import json
from datetime import datetime

def current_config():
    return dict(
        pde=st.session_state.pde,
        Nx=int(st.session_state.Nx), Ny=int(st.session_state.Ny),
        dtype=st.session_state.dtype,
        bc_left=st.session_state.bc_left, bc_right=st.session_state.bc_right,
        bc_bottom=st.session_state.bc_bottom, bc_top=st.session_state.bc_top,
        # Poisson
        poisson_tol=float(st.session_state.poisson_tol),
        poisson_maxiter=int(st.session_state.poisson_maxiter),
        # Diffusion
        diff_alpha=float(st.session_state.diff_alpha),
        diff_cfl=float(st.session_state.diff_cfl),
        diff_steps_per_frame=int(st.session_state.diff_steps_per_frame),
        diff_use_implicit=bool(st.session_state.diff_use_implicit),
        diff_dt_implicit=float(st.session_state.diff_dt_implicit),
        # Wave
        wave_c=float(st.session_state.wave_c),
        wave_cfl=float(st.session_state.wave_cfl),
        wave_steps_per_frame=int(st.session_state.wave_steps_per_frame),
        # Mask
        use_mask=bool(st.session_state.use_mask),
        phi_mask=float(st.session_state.phi_mask),
        mask_arr=(st.session_state.mask_arr.astype(int).tolist()
                  if (st.session_state.use_mask and st.session_state.mask_arr is not None) else None),
    )

def load_config(cfg: dict):
    ss = st.session_state
    for key in [
        "pde", "Nx", "Ny", "dtype",
        "poisson_tol", "poisson_maxiter",
        "diff_alpha", "diff_cfl", "diff_steps_per_frame",
        "diff_use_implicit", "diff_dt_implicit",
        "wave_c", "wave_cfl", "wave_steps_per_frame",
        "use_mask", "phi_mask",
    ]:
        if key in cfg:
            setattr(ss, key, cfg[key])
    for key in ["bc_left", "bc_right", "bc_bottom", "bc_top"]:
        if key in cfg:
            setattr(ss, key, tuple(cfg[key]))
    if cfg.get("mask_arr") is not None:
        arr = np.array(cfg["mask_arr"], dtype=bool)
        ss.mask_arr = arr
        ss.use_mask = True
    st.success("Config loaded. Click **Apply configuration** to rebuild the solver.")

st.markdown("### Config")
c1, c2 = st.columns(2)
with c1:
    if st.button("Save config"):
        cfg = current_config()
        data = json.dumps(cfg, indent=2).encode()
        st.download_button(
            "Download config.json",
            data=data,
            file_name=f"scfa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
with c2:
    cfg_file = st.file_uploader("Load config.json", type=["json"])
    if cfg_file is not None:
        try:
            cfg = json.loads(cfg_file.read().decode())
            load_config(cfg)
        except Exception as e:
            st.error(f"Failed to load config: {e}")

# ----------------- Export -----------------
st.markdown("### Export")
colE1, colE2 = st.columns(2)
if colE1.button("Export PNG"):
    png = export_png(st.session_state.field, title=f"{pde} field",
                     cmap="viridis" if pde!="Poisson" else "inferno")
    st.download_button("Download PNG", data=png,
                       file_name=f"{pde.lower()}_field.png", mime="image/png")
if colE2.button("Export .npz"):
    meta = dict(
        pde=pde, Nx=int(Nx), Ny=int(Ny), dtype=dtype,
        timestep=int(st.session_state.timestep),
        dt_info=st.session_state.dt_info,
        bc_left=st.session_state.bc_left, bc_right=st.session_state.bc_right,
        bc_bottom=st.session_state.bc_bottom, bc_top=st.session_state.bc_top,
        use_mask=bool(st.session_state.use_mask), phi_mask=float(st.session_state.phi_mask),
    )
    npz = export_npz(st.session_state.field, meta=meta)
    st.download_button("Download NPZ", data=npz,
                       file_name=f"{pde.lower()}_field.npz", mime="application/zip")

