// Scientific Computing for All — Stage 1 (GPU Core)
// =================================================
// A modular, (optionally) GPU-accelerated mini-framework for 2D PDEs in C++.
//
// Single-file implementation with CPU fallback. If compiled with NVCC and
// -DUSE_CUDA, a matrix-free 5-point Laplacian kernel is used for the
// homogeneous-Dirichlet path (float). All boundary constants (beta terms)
// are added to RHS/sources; the operator itself is strictly LINEAR.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <chrono>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace scfa {

// ------------------------------
// Utilities
// ------------------------------
inline double now_s() {
  using clock = std::chrono::high_resolution_clock;
  static const auto t0 = clock::now();
  auto t = clock::now();
  return std::chrono::duration<double>(t - t0).count();
}

struct Grid2D {
  int Nx{256};
  int Ny{256};
  double Lx{1.0};
  double Ly{1.0};
  double hx() const { return Lx / (Nx + 1.0); }
  double hy() const { return Ly / (Ny + 1.0); }
};

enum class BCKind { Dirichlet, Neumann, Robin };

struct BCSide {
  BCKind kind{BCKind::Dirichlet};
  // For Dirichlet/Neumann: value
  // For Robin: a,b,c stored in arr[0..2]
  std::array<double,3> arr{0.0,0.0,0.0};
};

struct Boundary {
  BCSide left, right, bottom, top;
};

inline std::pair<double,double> alpha_beta(const BCSide& s, double h) {
  if (s.kind == BCKind::Dirichlet) {
    double phi = s.arr[0];
    return {0.0, phi};
  } else if (s.kind == BCKind::Neumann) {
    double q = s.arr[0];
    return {1.0, h * q};
  } else { // Robin: a*u + b*du/dn = c
    double a=s.arr[0], b=s.arr[1], c=s.arr[2];
    if (std::abs(b) < 1e-30) {
      double phi = c / a; // Dirichlet fallback
      return {0.0, phi};
    }
    double alpha = 1.0 - (a * h) / b;
    double beta  = (h / b) * c;
    return {alpha, beta};
  }
}

inline bool is_homog_dirichlet(const Boundary& bc) {
  auto side_ok = [](const BCSide& s){
    if (s.kind != BCKind::Dirichlet) return false;
    return std::abs(s.arr[0]) < 1e-30;
  };
  return side_ok(bc.left) && side_ok(bc.right) && side_ok(bc.bottom) && side_ok(bc.top);
}

// Build RHS constants from wall betas, in units of Y = 4u - neighbors
inline std::vector<double> boundary_constant_vector(const Grid2D& g, const Boundary& bc) {
  int Ny = g.Ny, Nx = g.Nx; 
  double hx = g.hx(), hy = g.hy();
  std::vector<double> v(size_t(Nx)*Ny, 0.0);
  auto [aL,bL] = alpha_beta(bc.left,   hx);
  auto [aR,bR] = alpha_beta(bc.right,  hx);
  auto [aB,bB] = alpha_beta(bc.bottom, hy);
  auto [aT,bT] = alpha_beta(bc.top,    hy);
  if (std::abs(bL) != 0.0) for (int i=0;i<Ny;++i) v[size_t(i)*Nx + 0]      += bL;
  if (std::abs(bR) != 0.0) for (int i=0;i<Ny;++i) v[size_t(i)*Nx + (Nx-1)] += bR;
  if (std::abs(bB) != 0.0) for (int j=0;j<Nx;++j) v[size_t(0)*Nx + j]      += bB;
  if (std::abs(bT) != 0.0) for (int j=0;j<Nx;++j) v[size_t(Ny-1)*Nx + j]   += bT;
  return v;
}

// Mask helpers
inline std::vector<uint8_t> validate_mask(const std::vector<uint8_t>* mask, int Ny, int Nx) {
  if (!mask) return {};
  if (int(mask->size()) != Ny*Nx) throw std::runtime_error("mask size mismatch");
  return *mask;
}

inline std::vector<double> mask_constant_vector(const std::vector<uint8_t>& M, int Ny, int Nx, double phi_mask) {
  if (M.empty()) return {};
  std::vector<double> v(size_t(Nx)*Ny, 0.0);
  auto idx = [Nx](int i,int j){ return size_t(i)*Nx + j; };
  for (int i=0;i<Ny;++i){
    for (int j=0;j<Nx;++j){
      if (!M[idx(i,j)]) {
        // counts handled from neighbors; we only add constants on INSIDE cells that miss a neighbor
        continue;
      }
      // left neighbor outside
      if (j-1>=0 && !M[idx(i,j-1)]) v[idx(i,j)] += phi_mask;
      // right
      if (j+1<Nx && !M[idx(i,j+1)]) v[idx(i,j)] += phi_mask;
      // down (i+1)
      if (i+1<Ny && !M[idx(i+1,j)]) v[idx(i,j)] += phi_mask;
      // up (i-1)
      if (i-1>=0 && !M[idx(i-1,j)]) v[idx(i,j)] += phi_mask;
    }
  }
  return v;
}

inline void clamp_mask(std::vector<double>& u, const std::vector<uint8_t>& M, double phi) {
  if (M.empty()) return;
  for (size_t k=0;k<u.size();++k) if (!M[k]) u[k]=phi;
}

// ------------------------------
// CUDA kernel (optional)
// ------------------------------
#ifdef USE_CUDA
__global__ void laplace5_kernel(const float* __restrict__ u, float* __restrict__ y, int Nx, int Ny) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i>=Ny || j>=Nx) return;
  int idx = i * Nx + j;
  float uc = u[idx];
  float left  = (j>0   ) ? u[idx-1]   : 0.0f;
  float right = (j+1<Nx) ? u[idx+1]   : 0.0f;
  float up    = (i>0   ) ? u[idx-Nx]  : 0.0f;
  float down  = (i+1<Ny) ? u[idx+Nx]  : 0.0f;
  y[idx] = 4.0f*uc - left - right - up - down;
}

struct DeviceBuf {
  float* d_ptr{nullptr};
  size_t n{0};
  DeviceBuf()=default;
  explicit DeviceBuf(size_t n_):n(n_){ if(n){ cudaMalloc(&d_ptr, n*sizeof(float)); }}
  ~DeviceBuf(){ if(d_ptr) cudaFree(d_ptr); }
  DeviceBuf(const DeviceBuf&)=delete; DeviceBuf& operator=(const DeviceBuf&)=delete;
  DeviceBuf(DeviceBuf&& o) noexcept { d_ptr=o.d_ptr; n=o.n; o.d_ptr=nullptr; o.n=0; }
};
#endif

// ------------------------------
// Laplace5 operator (matrix-free, linear)
// ------------------------------
struct Laplace5 {
  Grid2D g;
  Boundary bc;
  bool use_cuda{false};
  bool homogD{false};
  std::pair<double,double> aL, aR, aB, aT; // (alpha,beta) per side (beta NOT used here)
  std::vector<uint8_t> mask; // 1=inside, 0=outside
  double phi_mask{0.0};

#ifdef USE_CUDA
  // Device buffers (reused) for matvec path
  mutable DeviceBuf d_u, d_y;
  dim3 gridDim{1,1,1}, blockDim{32,8,1};
#endif

  Laplace5(const Grid2D& grid, const Boundary& b, const std::vector<uint8_t>* mask_=nullptr, double phi_m=0.0, bool try_cuda=true)
  : g(grid), bc(b), phi_mask(phi_m) {
    mask = validate_mask(mask_, g.Ny, g.Nx);
    aL = alpha_beta(bc.left,   g.hx());
    aR = alpha_beta(bc.right,  g.hx());
    aB = alpha_beta(bc.bottom, g.hy());
    aT = alpha_beta(bc.top,    g.hy());
    homogD = is_homog_dirichlet(bc);
#ifdef USE_CUDA
    use_cuda = try_cuda && homogD && mask.empty();
    if (use_cuda) {
      size_t n = size_t(g.Nx)*g.Ny;
      d_u = DeviceBuf(n);
      d_y = DeviceBuf(n);
      int gx = (g.Nx + 32 - 1)/32;
      int gy = (g.Ny + 8  - 1)/8;
      gridDim = dim3(gx, gy, 1);
      blockDim= dim3(32, 8, 1);
    }
#else
    (void)try_cuda; use_cuda=false;
#endif
  }

  // y = A@v, where A returns Y = 4u - neighbors - alpha*edges (no betas), raveled
  void apply(const std::vector<double>& v, std::vector<double>& y) const {
    const int Nx=g.Nx, Ny=g.Ny; const size_t N=size_t(Nx)*Ny;
    y.assign(N, 0.0);

#ifdef USE_CUDA
    if (use_cuda) {
      // Only for float & homog Dirichlet & no mask; we cast to float
      std::vector<float> h_u(N), h_y(N);
      for (size_t k=0;k<N;++k) h_u[k] = static_cast<float>(v[k]);
      cudaMemcpy(d_u.d_ptr, h_u.data(), N*sizeof(float), cudaMemcpyHostToDevice);
      laplace5_kernel<<<gridDim, blockDim>>>(d_u.d_ptr, d_y.d_ptr, Nx, Ny);
      cudaMemcpy(h_y.data(), d_y.d_ptr, N*sizeof(float), cudaMemcpyDeviceToHost);
      for (size_t k=0;k<N;++k) y[k] = static_cast<double>(h_y[k]);
      // Linear BC alphas are zero for homog Dirichlet, so nothing else to do
      return;
    }
#endif
    auto idx = [Nx](int i,int j){ return size_t(i)*Nx + j; };

    const bool haveMask = !mask.empty();
    // Start with central 4u term on inside cells (or all cells if no mask)
    if (!haveMask) {
      for (int i=0;i<Ny;++i) for (int j=0;j<Nx;++j) y[idx(i,j)] = 4.0 * v[idx(i,j)];
      // neighbors
      for (int i=0;i<Ny;++i){
        for (int j=0;j<Nx;++j){
          double vij = v[idx(i,j)]; (void)vij;
          if (i-1>=0) y[idx(i,j)] -= v[idx(i-1,j)]; // up
          if (i+1<Ny) y[idx(i,j)] -= v[idx(i+1,j)]; // down
          if (j-1>=0) y[idx(i,j)] -= v[idx(i,j-1)]; // left
          if (j+1<Nx) y[idx(i,j)] -= v[idx(i,j+1)]; // right
        }
      }
      // BC alphas on edges
      if (aL.first!=0.0) for (int i=0;i<Ny;++i) y[idx(i,0   )] -= aL.first * v[idx(i,0   )];
      if (aR.first!=0.0) for (int i=0;i<Ny;++i) y[idx(i,Nx-1)] -= aR.first * v[idx(i,Nx-1)];
      if (aB.first!=0.0) for (int j=0;j<Nx;++j) y[idx(0,j   )] -= aB.first * v[idx(0,j   )];
      if (aT.first!=0.0) for (int j=0;j<Nx;++j) y[idx(Ny-1,j)] -= aT.first * v[idx(Ny-1,j)];
      return;
    }

    // With mask: only inside cells contribute; outside rows -> identity scaled later by solver
    for (int i=0;i<Ny;++i){
      for (int j=0;j<Nx;++j){
        const size_t k = idx(i,j);
        if (!mask[k]) { y[k] = v[k] * (g.hx()*g.hx()); continue; }
        double acc = 4.0 * v[k];
        // neighbors counted only if also inside
        if (i-1>=0 && mask[idx(i-1,j)]) acc -= v[idx(i-1,j)];
        if (i+1<Ny && mask[idx(i+1,j)]) acc -= v[idx(i+1,j)];
        if (j-1>=0 && mask[idx(i,j-1)]) acc -= v[idx(i,j-1)];
        if (j+1<Nx && mask[idx(i,j+1)]) acc -= v[idx(i,j+1)];
        y[k] = acc;
      }
    }
    // linear BC alphas, applied only on inside-edge cells
    if (aL.first!=0.0) for (int i=0;i<Ny;++i){ size_t k=idx(i,0); if(mask[k]) y[k] -= aL.first*v[k]; }
    if (aR.first!=0.0) for (int i=0;i<Ny;++i){ size_t k=idx(i,Nx-1); if(mask[k]) y[k] -= aR.first*v[k]; }
    if (aB.first!=0.0) for (int j=0;j<Nx;++j){ size_t k=idx(0,j); if(mask[k]) y[k] -= aB.first*v[k]; }
    if (aT.first!=0.0) for (int j=0;j<Nx;++j){ size_t k=idx(Ny-1,j); if(mask[k]) y[k] -= aT.first*v[k]; }
  }
};

// ------------------------------
// Small BLAS helpers on host
// ------------------------------
inline double dot(const std::vector<double>& a, const std::vector<double>& b){
  double s=0.0; size_t n=a.size();
  for (size_t i=0;i<n;++i) s += a[i]*b[i];
  return s;
}
inline double nrm2(const std::vector<double>& a){ return std::sqrt(dot(a,a)); }
inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y){
  size_t n=x.size(); for(size_t i=0;i<n;++i) y[i]+=alpha*x[i];
}
inline void scal(double alpha, std::vector<double>& x){ for(double& v: x) v*=alpha; }
inline std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b){
  std::vector<double> r=a; axpy(1.0,b,r); return r;
}

// ------------------------------
// Poisson (CG)
// ------------------------------
struct Poisson2D {
  Grid2D g; Boundary bc; Laplace5 A; double scale; // scale=1/h^2 so scale*A ~ -∇²
  std::vector<uint8_t> mask; double phi_mask{0.0};

  Poisson2D(const Grid2D& grid, const Boundary& b, const std::vector<uint8_t>* mask_=nullptr, double phi_m=0.0)
    : g(grid), bc(b), A(grid,b,mask_,phi_m,true), scale(1.0/(grid.hx()*grid.hx())) {
      mask = validate_mask(mask_, g.Ny, g.Nx);
      phi_mask = phi_m;
    }

  std::vector<double> rhs(std::function<double(double,double)> f) const {
    int Ny=g.Ny, Nx=g.Nx; double h=g.hx();
    std::vector<double> F(size_t(Nx)*Ny, 0.0);
    auto idx=[Nx](int i,int j){ return size_t(i)*Nx+j; };
    for (int i=0;i<Ny;++i){
      for (int j=0;j<Nx;++j){
        double x = (j+1)*h, y=(i+1)*h;
        if (f) F[idx(i,j)] = f(x,y);
        else   F[idx(i,j)] = 2.0*M_PI*M_PI*std::sin(M_PI*x)*std::sin(M_PI*y);
      }
    }
    return F;
  }
  // Convenience overload so P.rhs() works without including <functional> in user code
  std::vector<double> rhs() const { return rhs(nullptr); }

  // CG with simple Jacobi preconditioner: Minv ~ 0.25 on interior laplace rows, 1.0 on identity rows
  std::pair<std::vector<double>,int> solve_cg(const std::vector<double>& b_field, double tol=1e-8, int maxiter=2000) const {
    const size_t N = b_field.size();
    // Assemble RHS: b + scale*bc_const + mask-edge constants + identity rows outside
    auto bc_const = boundary_constant_vector(g, bc);
    auto mc = mask_constant_vector(mask, g.Ny, g.Nx, phi_mask);
    std::vector<double> rhs = b_field; 
    if(!bc_const.empty()){ std::vector<double> tmp=bc_const; scal(scale,tmp); axpy(1.0,tmp,rhs); }
    if(!mc.empty()){ std::vector<double> tmp=mc; scal(scale,tmp); axpy(1.0,tmp,rhs); }
    if(!mask.empty()){
      for (size_t k=0;k<N;++k) if (!mask[k]) rhs[k] += phi_mask; // identity rows
    }

    // CG solve for scale*(A@u) = rhs  -> operator is K(v)=scale*(A@v)
    auto K = [this](const std::vector<double>& v){ std::vector<double> y; A.apply(v,y); scal(scale,y); return y; };

    std::vector<double> u(N,0.0), r(N), z(N), p(N);
    // r = rhs - K(u)=rhs initially
    r = rhs;
    // Jacobi: diag ~ 4 for Laplacian rows => Minv=0.25, identity rows diag ~ 1/scale so Minv=1
    for (size_t k=0;k<N;++k) z[k] = (!mask.empty() && !mask[k]) ? r[k] : 0.25*r[k];
    p = z;
    double rz = dot(r,z); double r0 = nrm2(r);
    int it=0;
    for (; it<maxiter; ++it){
      std::vector<double> Ap = K(p);
      double alpha = rz / dot(p,Ap);
      axpy(alpha, p, u);
      axpy(-alpha, Ap, r);
      if (nrm2(r) <= tol * (r0>0?r0:1.0)) break;
      for (size_t k=0;k<N;++k) z[k] = (!mask.empty() && !mask[k]) ? r[k] : 0.25*r[k];
      double rz_new = dot(r,z);
      double beta = rz_new / rz;
      for (size_t k=0;k<N;++k) p[k] = z[k] + beta*p[k];
      rz = rz_new;
    }
    return {u, 0};
  }
};

// ------------------------------
// Diffusion (explicit + implicit BE)
// ------------------------------
struct Diffusion2D {
  Grid2D g; Boundary bc; Laplace5 A; double alpha{1.0}; double h; double dt_stable; 
  std::vector<uint8_t> mask; double phi_mask{0.0};
  std::vector<double> bc_src; // in ∇² units

  Diffusion2D(const Grid2D& grid, const Boundary& b, double alpha_=1.0, const std::vector<uint8_t>* mask_=nullptr, double phi_m=0.0)
  : g(grid), bc(b), A(grid,b,mask_,phi_m,true), alpha(alpha_), h(grid.hx()), mask(validate_mask(mask_,grid.Ny,grid.Nx)), phi_mask(phi_m) {
    dt_stable = (h*h)/(4.0*alpha);
    auto bc_const = boundary_constant_vector(g, bc);
    bc_src = bc_const; double invh2 = 1.0/(h*h); for (auto& v: bc_src) v *= invh2;
  }

  // Explicit Euler step
  std::vector<double> step_explicit(const std::vector<double>& u, double dt, const std::vector<double>* source=nullptr) const {
    if (dt > dt_stable + 1e-12) throw std::runtime_error("dt too large for explicit Euler");
    // lap u
    std::vector<double> Au; A.apply(u, Au); // Y = -(h^2 ∇² u)
    std::vector<double> lap(Au.size()); double invh2 = 1.0/(h*h);
    for (size_t k=0;k<Au.size();++k) lap[k] = (-Au[k]) * invh2; // ∇² u

    std::vector<double> rhs = lap; // alpha*lap_u
    for (double& v: rhs) v *= alpha;
    for (size_t k=0;k<rhs.size();++k) rhs[k] += bc_src[k];
    if (source) for (size_t k=0;k<rhs.size();++k) rhs[k] += (*source)[k];

    std::vector<double> u_new = u; for (size_t k=0;k<u_new.size();++k) u_new[k] += dt * rhs[k];
    clamp_mask(u_new, mask, phi_mask);
    return u_new;
  }

  // Backward Euler via CG on (I - dt*alpha*∇²) u^{n+1} = u^n + dt*(bc_src + source)
  std::vector<double> step_implicit_be(const std::vector<double>& u, double dt, const std::vector<double>* source=nullptr) const {
    double h2 = h*h; double gamma = (dt*alpha)/h2; // scales A inside M(v) = v + gamma*(A@v)
    std::vector<double> rhs = u; // field units
    // add dt*(bc_src + source)
    for (size_t k=0;k<rhs.size();++k) rhs[k] += dt * bc_src[k];
    if (source) for (size_t k=0;k<rhs.size();++k) rhs[k] += dt * (*source)[k];

    // CG solve M v = rhs
    auto Mv = [this,gamma](const std::vector<double>& x){ std::vector<double> y; A.apply(x,y); for (auto& z: y) z = x[&z-&y[0]] + gamma * z; return y; };
    // Implement with in-place form (avoid lambda capture by index):
    auto Mv2 = [this,gamma](const std::vector<double>& x){
      std::vector<double> y; A.apply(x,y);
      for (size_t k=0;k<y.size();++k) y[k] = x[k] + gamma*y[k];
      return y;
    };

    const size_t N = rhs.size();
    std::vector<double> v(N,0.0), r(N), z(N), p(N);
    // r = rhs - M(0) = rhs
    r = rhs;
    // Jacobi Minv: diag ≈ 1 + 4*gamma on interior rows, 1 on identity rows (outside mask)
    for (size_t k=0;k<N;++k) {
      double diag = (!mask.empty() && !mask[k]) ? 1.0 : (1.0 + 4.0*gamma);
      z[k] = r[k] / diag;
    }
    p = z; double rz = dot(r,z); double r0=nrm2(r);
    for (int it=0; it<1000; ++it){
      std::vector<double> Ap = Mv2(p);
      double alpha_k = rz / dot(p,Ap);
      axpy(alpha_k, p, v);
      axpy(-alpha_k, Ap, r);
      if (nrm2(r) <= 1e-6 * (r0>0?r0:1.0)) break;
      for (size_t k=0;k<N;++k){ double diag = (!mask.empty() && !mask[k]) ? 1.0 : (1.0 + 4.0*gamma); z[k] = r[k]/diag; }
      double rz_new = dot(r,z); double beta = rz_new / rz;
      for (size_t k=0;k<N;++k) p[k] = z[k] + beta*p[k];
      rz = rz_new;
    }
    clamp_mask(v, mask, phi_mask);
    return v;
  }
};

// ------------------------------
// Wave (leapfrog)
// ------------------------------
struct Wave2D {
  Grid2D g; Boundary bc; Laplace5 A; double c{1.0}; double h; double dt_cfl; 
  std::vector<uint8_t> mask; double phi_mask{0.0};
  std::vector<double> bc_src; // in ∇² units

  Wave2D(const Grid2D& grid, const Boundary& b, double c_=1.0, const std::vector<uint8_t>* mask_=nullptr, double phi_m=0.0)
  : g(grid), bc(b), A(grid,b,mask_,phi_m,true), c(c_), h(grid.hx()), mask(validate_mask(mask_,grid.Ny,grid.Nx)), phi_mask(phi_m) {
    dt_cfl = h/(c*std::sqrt(2.0));
    auto bc_const = boundary_constant_vector(g, bc);
    bc_src = bc_const; double invh2 = 1.0/(h*h); for (auto& v: bc_src) v *= invh2;
  }

  std::vector<double> lap(const std::vector<double>& u) const {
    std::vector<double> Au; A.apply(u,Au); // Y = -(h^2 ∇² u)
    std::vector<double> L(Au.size()); double invh2=1.0/(h*h);
    for (size_t k=0;k<Au.size();++k) L[k] = (-Au[k]) * invh2; return L;
  }

  std::vector<double> step_leapfrog(const std::vector<double>& u_curr, const std::vector<double>& u_prev, double dt, const std::vector<double>* source=nullptr) const {
    if (dt > dt_cfl + 1e-12) throw std::runtime_error("dt too large (CFL)");
    auto L = lap(u_curr);
    std::vector<double> extra = bc_src; if (source){ for (size_t k=0;k<extra.size();++k) extra[k]+=(*source)[k]; }
    std::vector<double> u_next(u_curr.size());
    double c2dt2 = c*c*dt*dt;
    for (size_t k=0;k<u_next.size();++k) u_next[k] = 2*u_curr[k] - u_prev[k] + c2dt2*L[k] + dt*dt*extra[k];
    clamp_mask(u_next, mask, phi_mask);
    return u_next;
  }
};

// ------------------------------
// Interactive Runner
// ------------------------------
static int prompt_int(const std::string& msg, int def){ std::cout<<msg<<" [default "<<def<<"]: "; std::string s; std::getline(std::cin,s); if(s.empty()) return def; try{ return std::stoi(s);}catch(...){ return def; } }
static double prompt_double(const std::string& msg, double def){ std::cout<<msg<<" [default "<<def<<"]: "; std::string s; std::getline(std::cin,s); if(s.empty()) return def; try{ return std::stod(s);}catch(...){ return def; } }
static int prompt_choice(const std::string& title, const std::vector<std::string>& opts, int def){
  std::cout<<"\n"<<title<<"\n"; for (size_t i=0;i<opts.size();++i) std::cout<<"  "<<(i+1)<<") "<<opts[i]<<"\n"; std::cout<<"Choose [1-"<<opts.size()<<"] (default "<<(def+1)<<"): "; std::string s; std::getline(std::cin,s); if(s.empty()) return def; try{int k=std::stoi(s)-1; if(k>=0 && k<(int)opts.size()) return k;}catch(...){ } return def; }

static BCSide prompt_bc_side(const std::string& name){
  int kind = prompt_choice(name+" boundary type", {"Dirichlet","Neumann","Robin"}, 0);
  BCSide s; if(kind==0){ s.kind=BCKind::Dirichlet; s.arr[0]=prompt_double(name+" Dirichlet value φ",0.0);} 
  else if(kind==1){ s.kind=BCKind::Neumann; s.arr[0]=prompt_double(name+" Neumann flux q (∂u/∂n)",0.0);} 
  else { s.kind=BCKind::Robin; s.arr[0]=prompt_double(name+" Robin a",1.0); s.arr[1]=prompt_double(name+" Robin b (≠0)",1.0); s.arr[2]=prompt_double(name+" Robin c",0.0);} 
  return s; }

void run_interactive(){
  std::cout<<"\n=== Scientific Computing for All — Stage 1 (C++/CUDA) ===\n";
#ifdef USE_CUDA
  std::cout<<"GPU laplacian kernel: enabled (homog Dirichlet only)\n";
#else
  std::cout<<"GPU laplacian kernel: disabled (CPU build)\n";
#endif

  int Nx = prompt_int("Grid Nx (columns)", 256);
  int Ny = prompt_int("Grid Ny (rows)", 256);
  int dtype_idx = prompt_choice("Numeric precision (runtime scalar)", {"float32 (faster)", "float64 (more precise)"}, 0);
  (void)dtype_idx; // Implementation uses double internally; you can switch to float if desired.

  int eq = prompt_choice("Which equation?", {"Poisson (steady)", "Diffusion (transient)", "Wave (leapfrog)"}, 0);

  std::cout<<"\n-- Boundary conditions --\n";
  Boundary bc; bc.left=prompt_bc_side("Left"); bc.right=prompt_bc_side("Right"); bc.bottom=prompt_bc_side("Bottom"); bc.top=prompt_bc_side("Top");
  Grid2D g{Nx,Ny,1.0,1.0};

  if (eq==0){
    double tol = prompt_double("CG tolerance (smaller is stricter)", 1e-6);
    int maxit = prompt_int("CG max iterations", 2000);
    Poisson2D P(g, bc);
    auto b = P.rhs();
    double t0=now_s(); auto [u,info] = P.solve_cg(b,tol,maxit); double t1=now_s(); (void)info;
    double rel_err = -1.0;
    if (is_homog_dirichlet(bc)){
      double h=g.hx(); double num=0.0, den=0.0; 
      for (int i=0;i<Ny;++i){ for(int j=0;j<Nx;++j){ double x=(j+1)*h, y=(i+1)*h; double ue=std::sin(M_PI*x)*std::sin(M_PI*y); double diff=u[size_t(i)*Nx+j]-ue; num+=diff*diff; den+=ue*ue; }}
      rel_err = std::sqrt(num/den);
    }
    std::cout<<"\nPoisson results:\n  time: "<<(t1-t0)<<" s | backend: "<<(is_homog_dirichlet(bc)?"GPU-kernel or CPU":"CPU")<<"\n  rel L2 error: ";
    if (rel_err>=0.0) std::cout<<rel_err<<"\n"; else std::cout<<"(no exact comparison for these BCs)\n";
  }
  else if (eq==1){
    double alpha = prompt_double("Diffusivity alpha", 1.0);
    int steps = prompt_int("Number of time steps", 400);
    double safety = prompt_double("CFL fraction (<=1.0)", 0.9);
    Diffusion2D D(g, bc, alpha);
    std::vector<double> u(size_t(Nx)*Ny, 0.0);
    int r = std::max(2, std::min(Nx,Ny)/50);
    for (int i=Ny/2 - r; i<Ny/2 + r; ++i) for (int j=Nx/2 - r; j<Nx/2 + r; ++j) u[size_t(i)*Nx+j]=1.0;
    double dt = safety * D.dt_stable;
    std::cout<<"Using dt="<<dt<<" (stable <= "<<D.dt_stable<<")\n";
    double t0=now_s();
    for (int s=0;s<steps;++s) u = D.step_explicit(u, dt);
    double t1=now_s();
    std::cout<<"\nDiffusion results:\n  steps: "<<steps<<" | dt: "<<dt<<" | time: "<<(t1-t0)<<" s\n";
    // (no plot)
  }
  else {
    double c = prompt_double("Wave speed c", 1.0);
    int steps = prompt_int("Number of time steps", 600);
    double safety = prompt_double("CFL fraction (<=1.0)", 0.9);
    Wave2D W(g, bc, c);
    double dt = safety * W.dt_cfl; std::cout<<"Using dt="<<dt<<" (CFL <= "<<W.dt_cfl<<")\n";
    // Gaussian bump IC
    double h=g.hx(); std::vector<double> u0(size_t(Nx)*Ny,0.0);
    for (int i=0;i<Ny;++i){ for(int j=0;j<Nx;++j){ double x=(j+1)*h, y=(i+1)*h; double dx=x-0.5, dy=y-0.5; double sigma=0.10; u0[size_t(i)*Nx+j]=std::exp(-(dx*dx+dy*dy)/(2*sigma*sigma)); }}
    // start-up step for zero initial velocity
    auto L0 = W.lap(u0); std::vector<double> u_prev=u0, u_curr=u0; double c2dt2=c*c*dt*dt; for(size_t k=0;k<u_prev.size();++k) u_prev[k] = u0[k] + 0.5*c2dt2*L0[k];
    double t0=now_s();
    for (int s=0;s<steps;++s){ auto u_next = W.step_leapfrog(u_curr, u_prev, dt); u_prev.swap(u_curr); u_curr.swap(u_next); }
    double t1=now_s();
    std::cout<<"\nWave results:\n  steps: "<<steps<<" | dt: "<<dt<<" | time: "<<(t1-t0)<<" s\n";
  }
}

} // namespace scfa

int main(){
  try{ scfa::run_interactive(); }
  catch(const std::exception& e){ std::cerr<<"Error: "<<e.what()<<"\n"; return 1; }
  return 0;
}

