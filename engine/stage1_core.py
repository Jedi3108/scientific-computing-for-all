"""
Scientific Computing for All — Stage 1 (GPU Core)
=================================================
A modular, GPU-accelerated mini-framework for 2D PDEs.

Stage 1 includes:
- Dirichlet, Neumann, and Robin boundary conditions (per side)
- Matrix-free GPU operators (5-point Laplacian) with CuPy RawKernels
  (auto-enabled for homogeneous Dirichlet; vectorized path otherwise)
- Equations: Poisson (steady), Diffusion (transient), Wave (leapfrog)
- CPU fallback (NumPy) so it runs anywhere
- Interactive menu runner

This version patches the operator to be strictly LINEAR.
All boundary constants (β) are added to RHS / sources, not inside A@u.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass

# ------------------------------
# Backend (CuPy if available)
# ------------------------------
use_cpu_fallback = bool(int(os.environ.get("CPU_FALLBACK", "0")))
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cxl
    xp = cp
    gpu = (not use_cpu_fallback)
except Exception:
    import numpy as xp
    cp = None
    cxl = None
    gpu = False

# ------------------------------
# Core data structures
# ------------------------------
@dataclass
class Grid2D:
    Nx: int
    Ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    @property
    def hx(self) -> float:
        return self.Lx/(self.Nx+1)

    @property
    def hy(self) -> float:
        return self.Ly/(self.Ny+1)

    @property
    def shape(self):
        return (self.Ny, self.Nx)

@dataclass
class Boundary:
    # Each side is a tuple: ("Dirichlet", phi) OR ("Neumann", q) OR ("Robin", (a,b,c))
    left:   tuple[str, float | tuple] = ("Dirichlet", 0.0)
    right:  tuple[str, float | tuple] = ("Dirichlet", 0.0)
    bottom: tuple[str, float | tuple] = ("Dirichlet", 0.0)
    top:    tuple[str, float | tuple] = ("Dirichlet", 0.0)

# ------------------------------
# Utility helpers
# ------------------------------
def arr_like(grid: Grid2D, dtype):
    return xp.zeros(grid.shape, dtype=dtype)

def to_backend(a):
    if gpu and not isinstance(a, xp.ndarray):
        return xp.asarray(a)
    return a

def _is_homog_dirichlet(bc: Boundary) -> bool:
    def side_ok(side):
        kind, val = side
        return (kind.lower() == "dirichlet") and (float(val) == 0.0)
    return side_ok(bc.left) and side_ok(bc.right) and side_ok(bc.bottom) and side_ok(bc.top)

def _alpha_beta(side, h):
    """Return (alpha, beta) such that u_ghost = alpha * u_edge + beta."""
    kind, val = side
    k = kind.lower()
    if k == "dirichlet":
        phi = float(val)
        return 0.0, float(phi)
    if k == "neumann":
        q = float(val)
        return 1.0, float(h*q)
    if k == "robin":
        a, b, c = val
        a = float(a); b = float(b); c = float(c)
        if abs(b) < 1e-30:
            # falls back to Dirichlet: a*u = c  -> u = c/a
            phi = c/a
            return 0.0, float(phi)
        alpha = 1.0 - (a*h)/b
        beta  = (h/b)*c
        return float(alpha), float(beta)
    raise ValueError(f"Unknown BC kind: {kind}")

def boundary_constant_vector(grid: Grid2D, bc: Boundary, dtype):
    """
    Build the additive RHS/source vector from the constant (beta) terms
    introduced by ghost-node elimination at boundaries.

    NOTE: Our operator returns Y = 4u - u_left - u_right - u_up - u_down,
    which equals -(h^2 ∇² u). The solver multiplies by scale=1/h^2 to recover -∇² u.
    Here we build the constant vector in the SAME units as 'Y', i.e., it contributes
    exactly like those missing neighbors in the discrete stencil (before scaling).
    """
    Ny, Nx = grid.Ny, grid.Nx
    hx, hy = grid.hx, grid.hy
    v = xp.zeros((Ny, Nx), dtype=dtype)

    (_, bL) = _alpha_beta(bc.left,   hx)
    (_, bR) = _alpha_beta(bc.right,  hx)
    (_, bB) = _alpha_beta(bc.bottom, hy)
    (_, bT) = _alpha_beta(bc.top,    hy)

    # Missing neighbors are subtracted in Y, so constants appear as "+ beta" on RHS
    # (move them to the right-hand side).
    if bL != 0.0:
        v[:, 0]  += bL
    if bR != 0.0:
        v[:, -1] += bR
    if bB != 0.0:
        v[0, :]  += bB
    if bT != 0.0:
        v[-1, :] += bT

    return v.ravel()

# ------------------------------
# GPU RawKernel: 5-point Laplacian (float32) — homogeneous Dirichlet only
# ------------------------------
if gpu:
    _laplace5_src = r"""
    extern "C" __global__ void laplace5(
        const float* __restrict__ u,
        float* __restrict__ y,
        const int Nx, const int Ny)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x; // x (columns)
        int i = blockIdx.y * blockDim.y + threadIdx.y; // y (rows)
        if (i >= Ny || j >= Nx) return;
        int idx = i * Nx + j;
        float uc = u[idx];
        float left  = (j>0   ) ? u[idx-1]   : 0.0f;
        float right = (j+1<Nx) ? u[idx+1]   : 0.0f;
        float up    = (i>0   ) ? u[idx-Nx]  : 0.0f;
        float down  = (i+1<Ny) ? u[idx+Nx]  : 0.0f;
        // Returns 4u - neighbors = -(h^2 * Laplacian(u)) in discrete form (no scaling).
        y[idx] = 4.0f*uc - left - right - up - down;
    }
    """
    _laplace5 = cp.RawKernel(_laplace5_src, "laplace5")

# ------------------------------
# Operators (matrix-free, strictly linear)
# ------------------------------
class Laplace5:
    """
    Matrix-free 5-point Laplacian with general BC via ghost elimination.
    Acts on interior unknowns arranged as Ny×Nx.

    IMPORTANT: This operator is STRICTLY LINEAR. It applies ONLY the alpha*u_edge
    parts of ghost-node substitution inside A@u. All constant beta terms are excluded
    here and must be added to RHS/source via boundary_constant_vector() at solver level.

    Returns Y = (4u - neighbors - alpha*edge_corrections) which equals -(h^2 ∇² u) up to scaling.
    """
    def __init__(self, grid: Grid2D, bc: Boundary, dtype="float32", use_kernel=True):
        self.g = grid
        self.bc = bc
        self.dtype = xp.float32 if str(dtype).startswith("float32") else xp.float64
        self.hx = grid.hx
        self.hy = grid.hy
        self.use_kernel = bool(use_kernel and gpu and self.dtype == xp.float32 and _is_homog_dirichlet(bc))
        if self.use_kernel:
            self.bx, self.by = 32, 8
            self.gx = (self.g.Nx + self.bx - 1)//self.bx
            self.gy = (self.g.Ny + self.by - 1)//self.by
        # Precompute (alpha, beta) per side (we will ONLY use alpha inside A, betas go to RHS)
        self.sides = {
            "left":   _alpha_beta(self.bc.left,   self.hx),
            "right":  _alpha_beta(self.bc.right,  self.hx),
            "bottom": _alpha_beta(self.bc.bottom, self.hy),
            "top":    _alpha_beta(self.bc.top,    self.hy),
        }

    def __matmul__(self, v: xp.ndarray) -> xp.ndarray:
        v = to_backend(v)
        Ny, Nx = self.g.Ny, self.g.Nx
        u = v.reshape(Ny, Nx)
        y = xp.empty_like(u)
        if self.use_kernel:
            _laplace5((self.gx, self.gy), (self.bx, self.by),
                      (u.ravel(), y.ravel(), xp.int32(Nx), xp.int32(Ny)))
            return y.ravel()

        # Vectorized linear path (CPU or GPU) with BC alphas only (NO betas)
        y[:] = 4.0 * u
        # interior neighbors
        y[1:,   :] -= u[:-1,  :]  # up
        y[:-1,  :] -= u[1:,   :]  # down
        y[:, 1: ] -= u[:, :-1]    # left
        y[:, :-1] -= u[:, 1: ]    # right

        # --- linear BC corrections via u_ghost = alpha*u_edge + beta   (use ONLY alpha part here)
        (aL, _bL) = self.sides["left"]
        (aR, _bR) = self.sides["right"]
        (aB, _bB) = self.sides["bottom"]
        (aT, _bT) = self.sides["top"]

        # left boundary affects column j=0 (missing left neighbor)
        if aL != 0.0:
            y[:, 0]  -= (aL * u[:, 0])
        # right boundary affects column j=Nx-1 (missing right neighbor)
        if aR != 0.0:
            y[:, -1] -= (aR * u[:, -1])
        # bottom boundary affects row i=0 (missing up neighbor)
        if aB != 0.0:
            y[0, :]  -= (aB * u[0, :])
        # top boundary affects row i=Ny-1 (missing down neighbor)
        if aT != 0.0:
            y[-1, :] -= (aT * u[-1, :])

        return y.ravel()

# ------------------------------
# Poisson solver (CG)  -- uses RHS += scale * bc_const
# ------------------------------
class Poisson2D:
    def __init__(self, grid: Grid2D, bc: Boundary, dtype="float32"):
        self.g = grid
        self.bc = bc
        self.dtype = xp.float32 if str(dtype).startswith("float32") else xp.float64
        self.A = Laplace5(grid, bc, dtype=dtype)
        # A returns Y = -(h^2 ∇² u); scale by 1/h^2 so scale*A = -∇²
        self.scale = 1.0/(self.g.hx*self.g.hx)  # assuming hx=hy

    def rhs(self, f_callable=None):
        Ny, Nx = self.g.Ny, self.g.Nx
        h = self.g.hx
        y = xp.arange(1, Ny+1, dtype=self.dtype)*h
        x = xp.arange(1, Nx+1, dtype=self.dtype)*h
        X, Y = xp.meshgrid(x, y, indexing="xy")
        if f_callable is None:
            # u = sin(pi x) sin(pi y) solves -∇² u = f
            F = 2*(xp.pi**2)*xp.sin(xp.pi*X)*xp.sin(xp.pi*Y)
        else:
            F = to_backend(f_callable(X, Y)).astype(self.dtype)
        return F  # (units of -∇² u)

    def solve_cg(self, b: xp.ndarray, tol=1e-8, maxiter=2000):
        # Add boundary constant vector on RHS (scaled to -∇² units)
        bc_const = boundary_constant_vector(self.g, self.bc, self.dtype)  # units of Y = -(h^2 ∇² u)
        b_eff = b.ravel() + self.scale * bc_const                         # convert to -∇² units

        n = self.g.Nx*self.g.Ny
        if gpu and cxl is not None:
            Aop = cxl.LinearOperator((n, n), matvec=lambda v: self.scale*(self.A@v), dtype=self.dtype)
            Minv = cxl.LinearOperator((n, n), matvec=lambda v: 0.25*v, dtype=self.dtype)
            u, info = cxl.cg(Aop, b_eff, tol=tol, maxiter=maxiter, M=Minv)
            return u.reshape(self.g.shape), info

        # CPU fallback (simple CG)
        u = xp.zeros(n, dtype=self.dtype)
        r = b_eff - self.scale*(self.A@u)
        z = 0.25*r
        p = z.copy()
        rz = float(xp.dot(r, z))
        r0 = float(xp.linalg.norm(r))
        for _ in range(1, maxiter+1):
            Ap = self.scale*(self.A@p)
            alpha = rz/float(xp.dot(p, Ap))
            u += alpha*p
            r -= alpha*Ap
            if float(xp.linalg.norm(r)) <= tol*(r0 if r0>0 else 1.0):
                break
            z = 0.25*r
            rz_new = float(xp.dot(r, z))
            beta = rz_new/rz
            p = z + beta*p
            rz = rz_new
        return u.reshape(self.g.shape), 0

# ------------------------------
# Diffusion solver (explicit Euler, CFL-safe)
# source includes + alpha * lap(u) + (scale * bc_const) term each step
# ------------------------------
class Diffusion2D:
    def __init__(self, grid: Grid2D, bc: Boundary, alpha=1.0, dtype="float32"):
        self.g = grid
        self.bc = bc
        self.alpha = float(alpha)
        self.dtype = xp.float32 if str(dtype).startswith("float32") else xp.float64
        self.A = Laplace5(grid, bc, dtype=dtype)
        self.h = grid.hx  # assume hx=hy
        # stability for explicit Euler (2D 5-pt): dt <= h^2/(4*alpha)
        self.dt_stable = (self.h*self.h)/(4.0*self.alpha)
        # Precompute RHS constant source from boundaries, in ∇² units
        self.bc_src = (1.0/(self.h*self.h)) * boundary_constant_vector(self.g, self.bc, self.dtype)
    def step_implicit_be(self, u: xp.ndarray, dt: float, source: xp.ndarray | None = None):
        """
        Backward Euler: (I - dt*alpha*∇²) u^{n+1} = u^n + dt*(bc_src + source)
        Works with masks and GPU CG. No CFL restriction (choose dt for accuracy).
        """
        u = to_backend(u)
        h2 = self.h * self.h
        gamma = (dt * self.alpha) / h2  # scales A@v (A returns -(h^2 ∇² v))

        # Right-hand side in field units
        rhs = u.ravel() + dt * self.bc_src.ravel()
        if source is not None:
            rhs = rhs + dt * to_backend(source).ravel()

        n = self.g.Nx * self.g.Ny

        # Linear operator: M(v) = v + gamma * (A@v)
        if gpu and cxl is not None:
            M = cxl.LinearOperator(
                (n, n),
                matvec=lambda v: (to_backend(v) + gamma * (self.A @ to_backend(v))).astype(self.dtype),
                dtype=self.dtype,
            )
            # Simple Jacobi on 5-pt => diag ≈ 1 + 4*gamma
            Minv = cxl.LinearOperator(
                (n, n),
                matvec=lambda v: (v / (1.0 + 4.0 * gamma)).astype(self.dtype),
                dtype=self.dtype,
            )
            u_new, info = cxl.cg(M, rhs, tol=1e-6, maxiter=500, M=Minv)
            u_new = u_new.reshape(self.g.shape)
        else:
            # CPU fallback CG (few lines, reusing same operator)
            v = xp.zeros(n, dtype=self.dtype)
            def Mv(x): return (x + gamma * (self.A @ x)).astype(self.dtype)
            r = rhs - Mv(v)
            z = r / (1.0 + 4.0 * gamma)
            p = z.copy()
            rz = float(xp.dot(r, z))
            r0 = float(xp.linalg.norm(r))
            for _ in range(1, 1000):
                Ap = Mv(p)
                alpha = rz / float(xp.dot(p, Ap))
                v += alpha * p
                r -= alpha * Ap
                if float(xp.linalg.norm(r)) <= 1e-6 * (r0 if r0 > 0 else 1.0):
                    break
                z = r / (1.0 + 4.0 * gamma)
                rz_new = float(xp.dot(r, z))
                beta = rz_new / rz
                p = z + beta * p
                rz = rz_new
            u_new = v.reshape(self.g.shape)

        # keep your existing mask clamping behavior
        return _clamp_mask(u_new, getattr(self, "mask", None), getattr(self, "phi_mask", 0.0))

    def step_explicit(self, u: xp.ndarray, dt: float, source: xp.ndarray | None = None) -> xp.ndarray:
        # make sure inputs are on the backend
        u = to_backend(u)
        if source is not None:
            source = to_backend(source)

        assert dt <= self.dt_stable + 1e-12, "dt too large for explicit Euler (reduce dt or use implicit scheme)"
        Au = self.A @ u.ravel()          # Y = -(h^2 ∇² u)
        lap_u = (-Au) / (self.h * self.h)  # ∇² u

        rhs = self.alpha * lap_u + self.bc_src  # bc_src already in ∇² units
        if source is not None:
            rhs = rhs + source.ravel()

        return u + dt * rhs.reshape(self.g.shape)


# ------------------------------
# Wave equation (leapfrog)
# u_tt = c^2 ∇² u + s
# Add c^2 * dt^2 * bc_src each step (bc_src is ∇²-units)
# ------------------------------
class Wave2D:
    def __init__(self, grid: Grid2D, bc: Boundary, c=1.0, dtype="float32"):
        self.g = grid
        self.bc = bc
        self.c = float(c)
        self.dtype = xp.float32 if str(dtype).startswith("float32") else xp.float64
        self.A = Laplace5(grid, bc, dtype=dtype)
        self.h = grid.hx
        self.dt_cfl = self.h/(self.c*(2.0**0.5))
        # Precompute boundary source in ∇² units
        self.bc_src = (1.0/(self.h*self.h)) * boundary_constant_vector(self.g, self.bc, self.dtype)

    def lap(self, u):
        u = to_backend(u)
        Au = self.A @ u.ravel()                      # 1D
        return (-Au.reshape(self.g.shape)) / (self.h * self.h)  # <-- 2D

    def step_leapfrog(self, u_curr: xp.ndarray, u_prev: xp.ndarray, dt: float, source: xp.ndarray|None=None):
        u_curr = to_backend(u_curr)
        u_prev = to_backend(u_prev) 
        assert dt <= self.dt_cfl + 1e-12, "dt too large for leapfrog (CFL violated)"
        lap_u = self.lap(u_curr)
        extra = self.bc_src
        if source is not None:
            extra = extra + source.ravel()
        return 2*u_curr - u_prev + (self.c**2)*(dt*dt)*lap_u.reshape(self.g.shape) + (dt*dt)*extra.reshape(self.g.shape)

# ------------------------------
# Interactive Stage-1 Runner (menu-driven)
# ------------------------------
def _prompt_choice(title, options, default_idx=0):
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    raw = input(f"Choose [1-{len(options)}] (default {default_idx+1}): ").strip()
    if not raw:
        return default_idx
    try:
        k = int(raw) - 1
        if 0 <= k < len(options):
            return k
    except Exception:
        pass
    print("Invalid choice; using default.")
    return default_idx

def _prompt_int(msg, default):
    raw = input(f"{msg} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v
    except Exception:
        print("Invalid int; using default.")
        return default

def _prompt_float(msg, default):
    raw = input(f"{msg} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v
    except Exception:
        print("Invalid number; using default.")
        return default

def _prompt_bc(side_name):
    kind_idx = _prompt_choice(
        f"{side_name} boundary type",
        ["Dirichlet", "Neumann", "Robin"],
        0,
    )
    if kind_idx == 0:
        val = _prompt_float(f"{side_name} Dirichlet value φ", 0.0)
        return ("Dirichlet", val)
    elif kind_idx == 1:
        val = _prompt_float(f"{side_name} Neumann flux q (∂u/∂n)", 0.0)
        return ("Neumann", val)
    else:
        a = _prompt_float(f"{side_name} Robin 'a'", 1.0)
        b = _prompt_float(f"{side_name} Robin 'b' (≠0)", 1.0)
        c = _prompt_float(f"{side_name} Robin 'c'", 0.0)
        return ("Robin", (a, b, c))

def run_interactive():
    print("\n=== Scientific Computing for All — Stage 1 ===")
    print("GPU backend:", "available" if gpu else "CPU fallback")
    if gpu:
        print("(Tip: set CPU_FALLBACK=1 to force CPU mode)")

    Nx = _prompt_int("Grid Nx (columns)", 256)
    Ny = _prompt_int("Grid Ny (rows)", 256)
    dtype_idx = _prompt_choice(
        "Numeric precision",
        ["float32 (faster)", "float64 (more precise)"],
        0,
    )
    dtype = "float32" if dtype_idx == 0 else "float64"

    eq_idx = _prompt_choice(
        "Which equation?",
        ["Poisson (steady)", "Diffusion (transient)", "Wave (leapfrog)"],
        0,
    )

    print("\n-- Boundary conditions (each side can be different) --")
    left  = _prompt_bc("Left")
    right = _prompt_bc("Right")
    bottom= _prompt_bc("Bottom")
    top   = _prompt_bc("Top")
    bc = Boundary(left=left, right=right, bottom=bottom, top=top)

    g = Grid2D(Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0)

    try:
        import matplotlib.pyplot as plt
        HAVE_PLOT = True
    except Exception:
        HAVE_PLOT = False

    if eq_idx == 0:
        # --- Poisson ---
        tol = _prompt_float("CG tolerance (smaller is stricter)", 1e-6)
        maxiter = _prompt_int("CG max iterations", 2000)

        poisson = Poisson2D(g, bc, dtype=dtype)
        b = poisson.rhs()  # analytic RHS for sin-sin solution

        t0 = time.perf_counter()
        u, info = poisson.solve_cg(b, tol=tol, maxiter=maxiter)
        if gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        # If BCs are homogeneous Dirichlet (β=0), we can compare to exact
        if _is_homog_dirichlet(bc):
            h = g.hx
            y = xp.arange(1, g.Ny + 1, dtype=u.dtype) * h
            x = xp.arange(1, g.Nx + 1, dtype=u.dtype) * h
            X, Y = xp.meshgrid(x, y, indexing="xy")
            u_exact = xp.sin(xp.pi * X) * xp.sin(xp.pi * Y)
            rel_err = float(xp.linalg.norm(u - u_exact) / xp.linalg.norm(u_exact))
            err_msg = f"relative L2 error vs exact: {rel_err:.3e}"
        else:
            err_msg = "(no exact comparison for these BCs)"

        print("\nPoisson results:")
        print(f"  time: {t1 - t0:.3f} s  | backend: {'GPU' if gpu else 'CPU'} | info: {info}")
        print(" ", err_msg)

        if HAVE_PLOT:
            plt.figure(figsize=(6, 4))
            plt.imshow((u if not gpu else cp.asnumpy(u)), origin="lower",
                       cmap="inferno", aspect="auto", extent=[0, g.Lx, 0, g.Ly])
            plt.colorbar()
            plt.title("Poisson solution u(x,y)")
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout(); plt.show()
        else:
            print("(matplotlib not available; skipping plot)")

    elif eq_idx == 1:
        # --- Diffusion ---
        alpha = _prompt_float("Diffusivity alpha", 1.0)
        steps = _prompt_int("Number of time steps", 400)
        safety = _prompt_float("CFL fraction (<=1.0)", 0.9)

        diff = Diffusion2D(g, bc, alpha=alpha, dtype=dtype)
        u = arr_like(g, dtype=xp.float32 if dtype == "float32" else xp.float64)

        r = max(2, min(g.Nx, g.Ny) // 50)
        u[g.Ny // 2 - r : g.Ny // 2 + r, g.Nx // 2 - r : g.Nx // 2 + r] = 1.0
        dt = safety * diff.dt_stable

        print(f"Using dt={dt:.3e} (stable <= {diff.dt_stable:.3e})")

        t0 = time.perf_counter()
        for _ in range(1, steps + 1):
            u = diff.step_explicit(u, dt)
        if gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        print("\nDiffusion results:")
        print(f"  steps: {steps} | dt: {dt:.3e} | time: {t1 - t0:.3f} s  | backend: {'GPU' if gpu else 'CPU'}")

        if HAVE_PLOT:
            plt.figure(figsize=(6, 4))
            plt.imshow((u if not gpu else cp.asnumpy(u)),
                       origin="lower", cmap="viridis", aspect="auto",
                       extent=[0, g.Lx, 0, g.Ly])
            plt.colorbar()
            plt.title("Diffusion: u(x,y) at final time")
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout(); plt.show()
        else:
            print("(matplotlib not available; skipping plot)")

    else:
        # --- Wave equation (leapfrog) ---
        c_speed = _prompt_float("Wave speed c", 1.0)
        steps = _prompt_int("Number of time steps", 600)
        safety = _prompt_float("CFL fraction (<=1.0)", 0.9)

        wave = Wave2D(g, bc, c=c_speed, dtype=dtype)

        # Initial condition: Gaussian bump at center, zero initial velocity
        h = g.hx
        y = xp.arange(1, g.Ny + 1, dtype=xp.float32 if dtype == "float32" else xp.float64) * h
        x = xp.arange(1, g.Nx + 1, dtype=xp.float32 if dtype == "float32" else xp.float64) * h
        X, Y = xp.meshgrid(x, y, indexing="xy")
        sigma = 0.10
        u0 = xp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * sigma ** 2))

        dt = safety * wave.dt_cfl
        print(f"Using dt={dt:.3e} (CFL <= {wave.dt_cfl:.3e})")

        # Start-up step for leapfrog (zero initial velocity):
        # u_{-1} = u_0 + 0.5 * c^2 * dt^2 * ∇² u_0
        lap_u0 = wave.lap(u0)
        u_prev = u0 + 0.5 * (c_speed ** 2) * (dt ** 2) * lap_u0
        u_curr = u0.copy()

        t0 = time.perf_counter()
        for _ in range(steps):
            u_next = wave.step_leapfrog(u_curr, u_prev, dt)
            u_prev, u_curr = u_curr, u_next
        if gpu:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        print("\nWave equation results:")
        print(f"  steps: {steps} | dt: {dt:.3e} | time: {t1 - t0:.3f} s  | backend: {'GPU' if gpu else 'CPU'}")

        if HAVE_PLOT:
            plt.figure(figsize=(6, 4))
            plt.imshow((u_curr if not gpu else cp.asnumpy(u_curr)),
                       origin="lower", cmap="plasma", aspect="auto",
                       extent=[0, g.Lx, 0, g.Ly])
            plt.colorbar()
            plt.title("Wave: u(x,y) at final time")
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout(); plt.show()
        else:
            print("(matplotlib not available; skipping plot)")

# === PATCH: mask support for Stage-1 engine =================================
# Usage:
#   mask: Ny×Nx boolean array (True = inside domain, False = outside)
#   phi_mask: Dirichlet value outside mask (default 0.0)
#
# Poisson: rows outside mask behave like identity; RHS gets +phi_mask there.
# Diffusion/Wave: after each step we clamp u[~mask] = phi_mask.
# Mask edges add constant +beta terms exactly like wall-BCs, with beta=phi_mask.
# =============================================================================

def _validate_mask(mask, Ny, Nx):
    """Ensure mask is backend (xp) boolean array with correct shape."""
    if mask is None:
        return None
    if mask.shape != (Ny, Nx):
        raise ValueError(f"mask shape {mask.shape} must be (Ny, Nx)=({Ny},{Nx})")
    # Convert to backend array (NumPy -> CuPy if gpu) and force boolean dtype
    return xp.asarray(mask, dtype=bool)

def mask_constant_vector(mask, phi_mask, dtype):
    """β terms from mask edges (Dirichlet φ_mask). Shape (n,)."""
    if mask is None:
        return None
    Ny, Nx = mask.shape
    v = xp.zeros((Ny, Nx), dtype=dtype)

    # Use backend boolean arrays explicitly
    m = xp.zeros_like(mask, dtype=bool); m[:, 1:]  = mask[:, 1:]  & (~mask[:, :-1])
    if bool(m.any()): v[m] += float(phi_mask)

    m = xp.zeros_like(mask, dtype=bool); m[:, :-1] = mask[:, :-1] & (~mask[:, 1:])
    if bool(m.any()): v[m] += float(phi_mask)

    m = xp.zeros_like(mask, dtype=bool); m[1:, :]  = mask[1:, :]  & (~mask[:-1, :])
    if bool(m.any()): v[m] += float(phi_mask)

    m = xp.zeros_like(mask, dtype=bool); m[:-1, :] = mask[:-1, :] & (~mask[1:, :])
    if bool(m.any()): v[m] += float(phi_mask)

    return v.ravel()


# ---- modify Laplace5 to accept mask & phi_mask ----
_orig_Laplace5_init = Laplace5.__init__
_orig_Laplace5_matmul = Laplace5.__matmul__

def _Laplace5_init_mask(self, grid, bc, dtype="float32", use_kernel=True, mask=None, phi_mask=0.0):
    self.mask = _validate_mask(mask if mask is not None else None, grid.Ny, grid.Nx)
    self.phi_mask = float(phi_mask)
    _orig_Laplace5_init(self, grid, bc, dtype=dtype, use_kernel=use_kernel)

def _Laplace5_matmul_mask(self, v):
    Ny, Nx = self.g.Ny, self.g.Nx
    u = v.reshape(Ny, Nx)
    M = self.mask
    # If no mask: original behavior
    if M is None:
        return _orig_Laplace5_matmul(self, v)

    # vectorized, linear path with mask
    y = xp.zeros_like(u)
    # center term only on inside cells
    y += 4.0 * u * M

    # neighbor terms only when both center and neighbor are inside
    # left neighbors contribute to j>=1
    y[:, 1:] -= u[:, :-1] * (M[:, 1:] & M[:, :-1])
    # right neighbors
    y[:, :-1] -= u[:, 1:] * (M[:, :-1] & M[:, 1:])
    # up neighbors
    y[1:, :]  -= u[:-1, :] * (M[1:, :] & M[:-1, :])
    # down neighbors
    y[:-1, :] -= u[1:, :]  * (M[:-1, :] & M[1:, :])

    # linear wall-BC alpha terms, applied only on inside cells
    (aL, _), (aR, _), (aB, _), (aT, _) = (
        self.sides["left"], self.sides["right"], self.sides["bottom"], self.sides["top"]
    )
    if aL != 0.0:  y[:, 0]  -= (aL * u[:, 0])   * M[:, 0]
    if aR != 0.0:  y[:, -1] -= (aR * u[:, -1])  * M[:, -1]
    if aB != 0.0:  y[0, :]  -= (aB * u[0, :])   * M[0, :]
    if aT != 0.0:  y[-1, :] -= (aT * u[-1, :])  * M[-1, :]

    # rows outside mask -> identity row (scaled so that scale*(A@u)=u)
    if bool((~M).any()):
        y[~M] = u[~M] * (self.g.hx * self.g.hx)  # 1/scale

    return y.ravel()

Laplace5.__init__ = _Laplace5_init_mask
Laplace5.__matmul__ = _Laplace5_matmul_mask

# ---- patch Poisson/Diffusion/Wave constructors to accept mask/phi_mask ----
_orig_Poisson2D_init = Poisson2D.__init__
def _Poisson2D_init_mask(self, grid, bc, dtype="float32", mask=None, phi_mask=0.0):
    self.mask = _validate_mask(mask if mask is not None else None, grid.Ny, grid.Nx)
    self.phi_mask = float(phi_mask)
    _orig_Poisson2D_init(self, grid, bc, dtype=dtype)
    # rebuild A with mask
    self.A = Laplace5(grid, bc, dtype=dtype, use_kernel=True, mask=self.mask, phi_mask=self.phi_mask)

Poisson2D.__init__ = _Poisson2D_init_mask

_orig_Diffusion2D_init = Diffusion2D.__init__
def _Diffusion2D_init_mask(self, grid, bc, alpha=1.0, dtype="float32", mask=None, phi_mask=0.0):
    self.mask = _validate_mask(mask if mask is not None else None, grid.Ny, grid.Nx)
    self.phi_mask = float(phi_mask)
    _orig_Diffusion2D_init(self, grid, bc, alpha=alpha, dtype=dtype)
    self.A = Laplace5(grid, bc, dtype=dtype, use_kernel=True, mask=self.mask, phi_mask=self.phi_mask)

Diffusion2D.__init__ = _Diffusion2D_init_mask

_orig_Wave2D_init = Wave2D.__init__
def _Wave2D_init_mask(self, grid, bc, c=1.0, dtype="float32", mask=None, phi_mask=0.0):
    self.mask = _validate_mask(mask if mask is not None else None, grid.Ny, grid.Nx)
    self.phi_mask = float(phi_mask)
    _orig_Wave2D_init(self, grid, bc, c=c, dtype=dtype)
    self.A = Laplace5(grid, bc, dtype=dtype, use_kernel=True, mask=self.mask, phi_mask=self.phi_mask)

Wave2D.__init__ = _Wave2D_init_mask

# ---- adjust RHS/sources to include mask constants & identity rows ----
def _poisson_solve_cg_mask(self, b_field, tol=1e-8, maxiter=2000):
    # base RHS (Ny×Nx) -> ravel
    b = b_field.ravel()

    # wall-BC constants (you already include via boundary_constant_vector in your patched code)
    bc_const = boundary_constant_vector(self.g, self.bc, self.dtype)

    # mask-edge constants (+phi_mask per missing neighbor)
    mc = mask_constant_vector(self.mask, self.phi_mask, self.dtype) if self.mask is not None else None
    if mc is not None:
        bc_const = bc_const + mc if bc_const is not None else mc

    # outside rows identity RHS (+phi_mask)
    if self.mask is not None:
        id_rhs = xp.where(self.mask.ravel(), 0.0, self.phi_mask).astype(self.dtype)
    else:
        id_rhs = None

    # assemble final RHS for Aop = scale*(A@v)
    rhs = b
    if bc_const is not None:
        rhs = rhs + self.scale * bc_const
    if id_rhs is not None:
        rhs = rhs + id_rhs

    n = self.g.Nx * self.g.Ny
    if gpu and cxl is not None:
        Aop = cxl.LinearOperator((n, n), matvec=lambda v: self.scale*(self.A@v), dtype=self.dtype)
        Minv = cxl.LinearOperator((n, n), matvec=lambda v: 0.25*v, dtype=self.dtype)
        u, info = cxl.cg(Aop, rhs, tol=tol, maxiter=maxiter, M=Minv)
        return u.reshape(self.g.shape), info

    # CPU fallback
    u = xp.zeros(n, dtype=self.dtype)
    r = rhs - self.scale*(self.A@u)
    z = 0.25 * r
    p = z.copy()
    rz = float(xp.dot(r, z))
    r0 = float(xp.linalg.norm(r))
    for _ in range(1, maxiter+1):
        Ap = self.scale*(self.A@p)
        alpha = rz / float(xp.dot(p, Ap))
        u += alpha * p
        r -= alpha * Ap
        if float(xp.linalg.norm(r)) <= tol*(r0 if r0>0 else 1.0):
            break
        z = 0.25 * r
        rz_new = float(xp.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return u.reshape(self.g.shape), 0

Poisson2D.solve_cg = _poisson_solve_cg_mask

# clamp helpers for time-dependent solvers
def _clamp_mask(u, mask, phi):
    if mask is None:
        return u
    u = u.copy()
    u[~mask] = float(phi)
    return u

_orig_Diffusion2D_step = Diffusion2D.step_explicit
def _Diffusion2D_step_mask(self, u, dt, source=None):
    u_new = _orig_Diffusion2D_step(self, u, dt, source=source)
    return _clamp_mask(u_new, self.mask, self.phi_mask)

Diffusion2D.step_explicit = _Diffusion2D_step_mask

_orig_Wave2D_step = Wave2D.step_leapfrog
def _Wave2D_step_mask(self, u_curr, u_prev, dt, source=None):
    u_next = _orig_Wave2D_step(self, u_curr, u_prev, dt, source=source)
    return _clamp_mask(u_next, self.mask, self.phi_mask)

Wave2D.step_leapfrog = _Wave2D_step_mask
# === END PATCH ===============================================================


if __name__ == "__main__":
    run_interactive()

