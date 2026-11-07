#Last Known Working Code That I fully understand -------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

# ------------------ SETUP ------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_PATH = "PINN_Bachelor_Research/Data/Cylinder flow data/cylinder_flow.npz"
SAVE_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Model Saves"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ LOAD + CROP ------------------
data = np.load(DATA_PATH)

u_grid = data["u"]
v_grid = data["v"]

ny, nx = u_grid.shape

crop_y, crop_x = 100, 100

y_start = max((ny - crop_y) // 2, 0)
y_end   = min(y_start + crop_y, ny)

x_start = 0
x_end   = min(crop_x, nx)

u_grid = u_grid[y_start:y_end, x_start:x_end]
v_grid = v_grid[y_start:y_end, x_start:x_end]

ny, nx = u_grid.shape

x_axis = np.arange(nx)
y_axis = np.arange(ny)
X, Y = np.meshgrid(x_axis, y_axis)

print(f"Cropped shape: ({ny}, {nx})")

X_MIN, X_MAX = x_axis.min(), x_axis.max()
Y_MIN, Y_MAX = y_axis.min(), y_axis.max()

# ------------------ GEOMETRY ------------------
cx, cy = 50, 50
r = 25

dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
cylinder = dist <= r

gap_thickness = 2          # in grid cells; g/D = gap_thickness / (2r)
R_outer = r + gap_thickness

gap_mask = (dist <= R_outer) & (~cylinder)

GD = gap_thickness / (2 * r)
RUN_TAG = f"gD{GD:.3f}"     # post-processing script must use the same tag
print("Run tag:", RUN_TAG)

# ------------------ NaN CLEANING ------------------
# Solid region: no-slip, zero is the correct fill.
u_grid = np.where(cylinder, 0.0, u_grid)
v_grid = np.where(cylinder, 0.0, v_grid)

# Leaked NaNs outside the cylinder: filled with the nearest valid fluid node so norms stay finite.
# They are excluded from training labels and from every metric via `ok`.
ok = np.isfinite(u_grid) & np.isfinite(v_grid)
print(f"NaN nodes outside cylinder: {(~ok).sum()} (of which in gap: {((~ok) & gap_mask).sum()})")

_, (_iy, _ix) = distance_transform_edt(~(ok & ~cylinder), return_indices=True)
u_grid = np.where(ok, u_grid, u_grid[_iy, _ix])
v_grid = np.where(ok, v_grid, v_grid[_iy, _ix])

fluid_eval = (~cylinder) & ok
gap_eval   = gap_mask & ok

# ------------------ TENSORS ------------------
def to_t(arr, grad_=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
    return t.requires_grad_(True) if grad_ else t

# GAP interior -> NS only (no velocity labels used)
X_gap_t = to_t(X[gap_mask], True)
Y_gap_t = to_t(Y[gap_mask], True)

# GAP BOUNDARY ring just outside R_outer -> data anchor
boundary_mask = (dist > R_outer) & (dist <= R_outer + 4.0) & ok

X_b_t = to_t(X[boundary_mask])
Y_b_t = to_t(Y[boundary_mask])
U_b_t = to_t(u_grid[boundary_mask])
V_b_t = to_t(v_grid[boundary_mask])

# DATA outside gap and ring
data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask) & ok

X_out_t = to_t(X[data_mask])
Y_out_t = to_t(Y[data_mask])
U_out_t = to_t(u_grid[data_mask])
V_out_t = to_t(v_grid[data_mask])

print(f"GAP interior (NS): {len(X_gap_t):,}")
print(f"GAP boundary (BC): {len(X_b_t):,}")
print(f"DATA (outside gap): {len(X_out_t):,}")

# Wall-biased NS collocation points (continuous sampling in the annulus,
# skewed toward the wall with rand**2)
N_wall = 10000
th_w = 2*np.pi*np.random.rand(N_wall)
rr_w = r + gap_thickness * np.random.rand(N_wall)**3
X_wall_t = to_t(cx + rr_w*np.cos(th_w), True)
Y_wall_t = to_t(cy + rr_w*np.sin(th_w), True)

# ------------------ HYPERPARAMS ------------------
h = 64

# Viscosity in GRID-INDEX units: nu = U_ref * D_idx / Re.
nu = 0.5

b_ns   = 2000
b_data = 10000
b_bc   = 5000
b_gap  = 1000

w_ns   = 250
w_bc   = 500
w_data = 500
w_gap  = 200

EPOCHS = 10001

# ------------------ MODEL ------------------
def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 3)
        )

    def forward(self, x, y):
        xn = normalise(x, X_MIN, X_MAX)
        yn = normalise(y, Y_MIN, Y_MAX)

        out = self.net(torch.cat([xn, yn], dim=1))
        u_raw, v_raw, p = out.split(1, dim=1)

        rho = torch.sqrt((x - cx)**2 + (y - cy)**2 + 1e-12)
        d = rho - r

        phi = d / (torch.sqrt(d**2 + 1e-12) + 1.0)

        u = phi * u_raw
        v = phi * v_raw

        return torch.cat([u, v, p], dim=1)


def grad(f, x):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]


model = PINN().to(device)

# ------------------ LOSSES ------------------
def NS_loss_fn(x, y):
    u, v, p = model(x, y).split(1, dim=1)

    ux, uy = grad(u, x), grad(u, y)
    vx, vy = grad(v, x), grad(v, y)

    uxx, uyy = grad(ux, x), grad(uy, y)
    vxx, vyy = grad(vx, x), grad(vy, y)

    px, py = grad(p, x), grad(p, y)

    cont = ux + vy
    momx = u*ux + v*uy - nu*(uxx + uyy) + px
    momy = u*vx + v*vy - nu*(vxx + vyy) + py

    return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)


def NS_loss():
    i1 = torch.randperm(len(X_gap_t),  device=device)[:b_ns//2]
    i2 = torch.randperm(len(X_wall_t), device=device)[:b_ns//2]
    return NS_loss_fn(X_gap_t[i1], Y_gap_t[i1]) + 3*NS_loss_fn(X_wall_t[i2], Y_wall_t[i2])


def data_loss():
    idx = torch.randperm(len(X_out_t), device=device)[:b_data]
    u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)
    return torch.mean((u_p - U_out_t[idx])**2) + torch.mean((v_p - V_out_t[idx])**2)


def bc_cylinder():
    a = 2 * np.pi * torch.rand(b_bc, 1, device=device)
    x = cx + r * torch.cos(a)
    y = cy + r * torch.sin(a)
    u, v, _ = model(x, y).split(1, dim=1)
    return torch.mean(u**2 + v**2)


def bc_gap():
    idx = torch.randperm(len(X_b_t), device=device)[:b_gap]
    u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)
    return torch.mean((u_p - U_b_t[idx])**2) + torch.mean((v_p - V_b_t[idx])**2)


def loss():
    ns = NS_loss()
    dat = data_loss()
    gap = bc_gap()

    total = w_ns * ns + w_data * dat + w_gap * gap

    return total, ns.item(), dat.item(), 0.0, gap.item()


# ------------------ TRAINING + HISTORY ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS//3, gamma=0.5)

history = {
    "epoch":       [],
    "loss_total":  [],
    "loss_ns":     [],
    "loss_data":   [],
    "loss_bc":     [],
    "loss_bc_gap": [],
}

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val, bcgap_val = loss()
    L.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 50 == 0:
        history["epoch"].append(epoch)
        history["loss_total"].append(L.item())
        history["loss_ns"].append(ns_val)
        history["loss_data"].append(dat_val)
        history["loss_bc"].append(bc_val)
        history["loss_bc_gap"].append(bcgap_val)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | Loss {L.item():.3e} | "
              f"NS {ns_val:.3e} | Data {dat_val:.3e} | "
              f"BC_gap {bcgap_val:.3e}")

# ------------------ SAVE MODEL + POST-PROCESSING DATA ------------------
# Save immediately after training so an interrupted post-processing run
# never requires retraining.
model.eval()

model_path = os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt")
post_path  = os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz")

torch.save(model.state_dict(), model_path)

np.savez(
    post_path,
    X=X, Y=Y,
    cylinder=cylinder, gap_mask=gap_mask, ok=ok,
    cx=cx, cy=cy, r=r, gap_thickness=gap_thickness,
    u_grid=u_grid, v_grid=v_grid,
    h=h, nu=nu,
    **{f"hist_{k}": np.array(v) for k, v in history.items()},
)

print(f"Saved model -> {model_path}")
print(f"Saved post-processing data -> {post_path}")
print("Training complete. All diagnostics are handled by the post-processing script.")

# # Same as above with little improvements and longer runs -------------------------------------------------------------------------------------------

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy.ndimage import distance_transform_edt

# # ------------------ SETUP ------------------
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# DATA_PATH = "PINN_Bachelor_Research/Data/Cylinder flow data/cylinder_flow.npz"
# SAVE_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Model Saves"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ------------------ LOAD + CROP ------------------
# data = np.load(DATA_PATH)

# u_grid = data["u"]
# v_grid = data["v"]

# ny, nx = u_grid.shape

# crop_y, crop_x = 100, 100

# y_start = max((ny - crop_y) // 2, 0)
# y_end   = min(y_start + crop_y, ny)

# x_start = 0
# x_end   = min(crop_x, nx)

# u_grid = u_grid[y_start:y_end, x_start:x_end]
# v_grid = v_grid[y_start:y_end, x_start:x_end]

# ny, nx = u_grid.shape

# x_axis = np.arange(nx)
# y_axis = np.arange(ny)
# X, Y = np.meshgrid(x_axis, y_axis)

# print(f"Cropped shape: ({ny}, {nx})")

# X_MIN, X_MAX = x_axis.min(), x_axis.max()
# Y_MIN, Y_MAX = y_axis.min(), y_axis.max()

# # ------------------ GEOMETRY ------------------
# cx, cy = 50, 50
# r = 25

# dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
# cylinder = dist <= r

# gap_thickness = 4          # in grid cells; g/D = gap_thickness / (2r)
# R_outer = r + gap_thickness

# gap_mask = (dist <= R_outer) & (~cylinder)

# GD = gap_thickness / (2 * r)
# RUN_TAG = f"gD{GD:.3f}OG"    # post-processing script must use the same tag
# print("Run tag:", RUN_TAG)

# # ------------------ NaN CLEANING ------------------
# # Solid region: no-slip, zero is the correct fill.
# u_grid = np.where(cylinder, 0.0, u_grid)
# v_grid = np.where(cylinder, 0.0, v_grid)

# # Leaked NaNs outside the cylinder: filled with the nearest valid fluid node so norms stay finite.
# # They are excluded from training labels and from every metric via `ok`.
# ok = np.isfinite(u_grid) & np.isfinite(v_grid)
# print(f"NaN nodes outside cylinder: {(~ok).sum()} (of which in gap: {((~ok) & gap_mask).sum()})")

# _, (_iy, _ix) = distance_transform_edt(~(ok & ~cylinder), return_indices=True)
# u_grid = np.where(ok, u_grid, u_grid[_iy, _ix])
# v_grid = np.where(ok, v_grid, v_grid[_iy, _ix])

# fluid_eval = (~cylinder) & ok
# gap_eval   = gap_mask & ok

# # ------------------ TENSORS ------------------
# def to_t(arr, grad_=False):
#     t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
#     return t.requires_grad_(True) if grad_ else t

# # GAP interior -> NS only (no velocity labels used)
# X_gap_t = to_t(X[gap_mask], True)
# Y_gap_t = to_t(Y[gap_mask], True)

# # GAP BOUNDARY ring just outside R_outer -> data anchor
# boundary_mask = (dist > R_outer) & (dist <= R_outer + 3.0) & ok

# X_b_t = to_t(X[boundary_mask])
# Y_b_t = to_t(Y[boundary_mask])
# U_b_t = to_t(u_grid[boundary_mask])
# V_b_t = to_t(v_grid[boundary_mask])

# # DATA outside gap and ring
# data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask) & ok

# X_out_t = to_t(X[data_mask])
# Y_out_t = to_t(Y[data_mask])
# U_out_t = to_t(u_grid[data_mask])
# V_out_t = to_t(v_grid[data_mask])

# print(f"GAP interior (NS): {len(X_gap_t):,}")
# print(f"GAP boundary (BC): {len(X_b_t):,}")
# print(f"DATA (outside gap): {len(X_out_t):,}")

# # Wall-biased NS collocation points (continuous sampling in the annulus,
# # skewed toward the wall with rand**2)
# N_wall = 10000
# th_w = 2*np.pi*np.random.rand(N_wall)
# rr_w = r + gap_thickness * np.random.rand(N_wall)**2
# X_wall_t = to_t(cx + rr_w*np.cos(th_w), True)
# Y_wall_t = to_t(cy + rr_w*np.sin(th_w), True)

# # ------------------ HYPERPARAMS ------------------
# h = 64

# # Viscosity in GRID-INDEX units: nu = U_ref * D_idx / Re  (U_ref = 1, D_idx = 50, Re = 100)
# nu = 0.5

# b_ns   = 2000
# b_data = 10000
# b_bc   = 5000
# b_gap  = 1000

# # fixed weights
# w_ns   = 250
# w_bc   = 500
# w_data = 100
# w_gap  = 50

# EPOCHS      = 15001      # Adam iterations (set 3001 to reproduce the old short run)
# LBFGS_ITERS = 3000       # full-batch L-BFGS iterations after Adam (0 = skip)
# LBFGS_CHUNK = 250        # L-BFGS iterations per optimiser call (progress is printed after each chunk)

# # ------------------ MODEL ------------------
# def normalise(val, vmin, vmax):
#     return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


# class PINN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2, h), nn.Tanh(),
#             nn.Linear(h, h), nn.Tanh(),
#             nn.Linear(h, h), nn.Tanh(),
#             nn.Linear(h, h), nn.Tanh(),
#             nn.Linear(h, 3)
#         )

#     def forward(self, x, y):
#         xn = normalise(x, X_MIN, X_MAX)
#         yn = normalise(y, Y_MIN, Y_MAX)
#         return self.net(torch.cat([xn, yn], dim=1))


# def grad(f, x):
#     return torch.autograd.grad(
#         f, x,
#         grad_outputs=torch.ones_like(f),
#         create_graph=True
#     )[0]


# model = PINN().to(device)

# # ------------------ LOSSES ------------------
# def NS_loss_fn(x, y):
#     u, v, p = model(x, y).split(1, dim=1)

#     ux, uy = grad(u, x), grad(u, y)
#     vx, vy = grad(v, x), grad(v, y)

#     uxx, uyy = grad(ux, x), grad(uy, y)
#     vxx, vyy = grad(vx, x), grad(vy, y)

#     px, py = grad(p, x), grad(p, y)

#     cont = ux + vy
#     momx = u*ux + v*uy - nu*(uxx + uyy) + px
#     momy = u*vx + v*vy - nu*(vxx + vyy) + py

#     return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)


# def NS_loss():
#     i1 = torch.randperm(len(X_gap_t),  device=device)[:b_ns//2]
#     i2 = torch.randperm(len(X_wall_t), device=device)[:b_ns//2]
#     return NS_loss_fn(X_gap_t[i1], Y_gap_t[i1]) + 5*NS_loss_fn(X_wall_t[i2], Y_wall_t[i2])


# def data_loss():
#     idx = torch.randperm(len(X_out_t), device=device)[:b_data]
#     u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)
#     return torch.mean((u_p - U_out_t[idx])**2) + torch.mean((v_p - V_out_t[idx])**2)


# def bc_cylinder():
#     a = 2 * np.pi * torch.rand(b_bc, 1, device=device)
#     x = cx + r * torch.cos(a)
#     y = cy + r * torch.sin(a)
#     u, v, _ = model(x, y).split(1, dim=1)
#     return torch.mean(u**2 + v**2)


# def bc_gap():
#     idx = torch.randperm(len(X_b_t), device=device)[:b_gap]
#     u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)
#     return torch.mean((u_p - U_b_t[idx])**2) + torch.mean((v_p - V_b_t[idx])**2)


# def loss():
#     ns  = NS_loss()
#     dat = data_loss()
#     bc  = bc_cylinder()
#     gap = bc_gap()
#     total = w_ns*ns + w_data*dat + w_bc*bc + w_gap*gap
#     return total, ns.item(), dat.item(), bc.item(), gap.item()


# # Deterministic full-batch version of the same loss (needed by L-BFGS: curvature estimates and the
# # line search break if the loss changes randomly between evaluations). Same terms, same weights.
# _a_fix = torch.linspace(0, 2 * np.pi, b_bc, device=device).unsqueeze(1)


# def loss_full():
#     ns = NS_loss_fn(X_gap_t, Y_gap_t) + 5 * NS_loss_fn(X_wall_t, Y_wall_t)
#     u_p, v_p, _ = model(X_out_t, Y_out_t).split(1, dim=1)
#     dat = torch.mean((u_p - U_out_t)**2) + torch.mean((v_p - V_out_t)**2)
#     u_c, v_c, _ = model(cx + r * torch.cos(_a_fix), cy + r * torch.sin(_a_fix)).split(1, dim=1)
#     bc = torch.mean(u_c**2 + v_c**2)
#     u_g, v_g, _ = model(X_b_t, Y_b_t).split(1, dim=1)
#     gap = torch.mean((u_g - U_b_t)**2) + torch.mean((v_g - V_b_t)**2)
#     total = w_ns*ns + w_data*dat + w_bc*bc + w_gap*gap
#     return total, ns.item(), dat.item(), bc.item(), gap.item()


# # ------------------ PHYSICS-ONLY DIAGNOSTIC (no ground truth) ------------------
# _gap_eval_idx = np.random.RandomState(SEED).choice(
#     len(X_gap_t), size=min(1000, len(X_gap_t)), replace=False)
# _Xe = X_gap_t.detach()[_gap_eval_idx].clone()
# _Ye = Y_gap_t.detach()[_gap_eval_idx].clone()

# def gap_residual():
#     x = _Xe.clone().requires_grad_(True)
#     y = _Ye.clone().requires_grad_(True)
#     return NS_loss_fn(x, y).item()


# # ------------------ HISTORY ------------------
# history = {
#     "epoch":       [],
#     "loss_total":  [],
#     "loss_ns":     [],
#     "loss_data":   [],
#     "loss_bc":     [],
#     "loss_bc_gap": [],
#     "gap_resid":   [],
#     "rel_l2_full": [],
#     "rel_l2_gap":  [],
# }

# X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
# Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

# U_true = np.sqrt(u_grid**2 + v_grid**2)


# def record(epoch, L_val, ns_val, dat_val, bc_val, bcgap_val):
#     """Ground truth is used for monitoring only, never for training in the gap."""
#     with torch.no_grad():
#         u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
#         U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()

#     rel_l2_full = np.linalg.norm((U_pred - U_true)[fluid_eval]) / np.linalg.norm(U_true[fluid_eval])
#     rel_l2_gap  = np.linalg.norm((U_pred - U_true)[gap_eval])   / np.linalg.norm(U_true[gap_eval])

#     history["epoch"].append(epoch)
#     history["loss_total"].append(L_val)
#     history["loss_ns"].append(ns_val)
#     history["loss_data"].append(dat_val)
#     history["loss_bc"].append(bc_val)
#     history["loss_bc_gap"].append(bcgap_val)
#     history["gap_resid"].append(gap_residual())
#     history["rel_l2_full"].append(rel_l2_full)
#     history["rel_l2_gap"].append(rel_l2_gap)


# # ------------------ TRAINING: ADAM ------------------
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS//3, gamma=0.5)

# for epoch in range(EPOCHS):
#     optimizer.zero_grad()
#     L, ns_val, dat_val, bc_val, bcgap_val = loss()
#     L.backward()
#     optimizer.step()
#     scheduler.step()

#     if epoch % 50 == 0:
#         record(epoch, L.item(), ns_val, dat_val, bc_val, bcgap_val)

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch:5d} | Loss {history['loss_total'][-1]:.3e} | "
#               f"NS {history['loss_ns'][-1]:.3e} | "
#               f"Data {history['loss_data'][-1]:.3e} | "
#               f"BC {history['loss_bc'][-1]:.3e} | "
#               f"BC_gap {history['loss_bc_gap'][-1]:.3e} | "
#               f"GapRes {history['gap_resid'][-1]:.3e} | "
#               f"L2 gap {history['rel_l2_gap'][-1]:.4e}")

# # ------------------ TRAINING: L-BFGS (full batch, fixed weights) ------------------
# if LBFGS_ITERS > 0:
#     print("\nL-BFGS ...")
#     lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_CHUNK,
#                               max_eval=int(1.25 * LBFGS_CHUNK), history_size=50,
#                               tolerance_grad=1e-10, tolerance_change=1e-14,
#                               line_search_fn="strong_wolfe")

#     def closure():
#         lbfgs.zero_grad()
#         L_ = loss_full()[0]
#         L_.backward()
#         return L_

#     n_chunks = max(LBFGS_ITERS // LBFGS_CHUNK, 1)
#     for k in range(n_chunks):
#         lbfgs.step(closure)
#         L_val, ns_val, dat_val, bc_val, bcgap_val = loss_full()
#         record(EPOCHS + (k + 1) * LBFGS_CHUNK, L_val.item(), ns_val, dat_val, bc_val, bcgap_val)
#         print(f"L-BFGS {(k + 1) * LBFGS_CHUNK:5d} | Loss {history['loss_total'][-1]:.3e} | "
#               f"NS {history['loss_ns'][-1]:.3e} | Data {history['loss_data'][-1]:.3e} | "
#               f"BC {history['loss_bc'][-1]:.3e} | BC_gap {history['loss_bc_gap'][-1]:.3e} | "
#               f"L2 gap {history['rel_l2_gap'][-1]:.4e} | L2 fluid {history['rel_l2_full'][-1]:.4e}")

# # ------------------ SAVE MODEL + POST-PROCESSING DATA ------------------
# model.eval()

# model_path = os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt")
# post_path  = os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz")

# torch.save(model.state_dict(), model_path)

# np.savez(
#     post_path,
#     X=X, Y=Y,
#     cylinder=cylinder, gap_mask=gap_mask, ok=ok,
#     cx=cx, cy=cy, r=r, gap_thickness=gap_thickness,
#     u_grid=u_grid, v_grid=v_grid,
#     h=h, nu=nu,
#     # metadata read by postprocess_dns.py (this plain network: 3 hidden layers, (x, y) inputs, soft no-slip)
#     n_hidden=3, r_wall=r, d_len=float(gap_thickness), hard_noslip=0, use_wall_feat=0,
#     **{f"hist_{k}": np.array(v) for k, v in history.items()},
# )

# print(f"Saved model -> {model_path}")
# print(f"Saved post-processing data -> {post_path}")


# Complex code with low error -----------------------------------------------------------------------------------------------------------------------

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy.ndimage import distance_transform_edt

# # =====================================================================
# # CONFIG
# # =====================================================================
# SEED = 42
# DATA_PATH = "PINN_Bachelor_Research/Data/Cylinder flow data/cylinder_flow.npz"
# SAVE_DIR = r"C:\Users\bhavy\Downloads\PINN_Research\PINN_Bachelor_Research\Code\FinalThesisCode"

# # --- data / geometry ---
# gap_thickness = 4            # [cells] measured from the GEOMETRIC cylinder surface r; g/D = gap_thickness / (2r)
# RING_WIDTH = 3.0             # [cells] thickness of the labelled data ring just outside the gap (value AND slope info)

# # --- wall model (immersed-boundary smearing) ---
# WALL_OFFSET = None           # [cells] effective no-slip radius = r + WALL_OFFSET. None = auto-calibrate from the data
#                              # (see calibrate_wall_offset); set a float to override, 0.0 = geometric wall.
# HARD_NOSLIP = False           # u,v = tanh(d/l) * net(...) for d>=0, exactly 0 inside the effective wall

# # --- network ---
# H = 128
# N_HIDDEN = 4

# # --- physics (Re = 100, U_inf = 1, D = 50 cells) ---
# nu = 0.5                     # = U_ref * D_idx / Re

# # --- loss weights (grid units: NS residual is ~1/D_idx of its D-unit size, hence the D_idx**2) ---
# D_idx = 50.0
# W_INIT = {"ns": D_idx ** 2, "data": 100.0, "bc": 500.0, "gap": 50.0}
# ADAPTIVE = True              # gradient-norm weight balancing (Wang et al.); w["ns"] stays fixed as the reference
# ADAPT_EVERY = 100
# ADAPT_ALPHA = 0.9

# # --- batches ---
# b_ns, b_data, b_bc, b_gap = 2000, 10000, 5000, 2000
# N_wall = 10000

# # --- optimisation ---
# EPOCHS_ADAM = 15001
# LR = 1e-3
# LBFGS_ITERS = 2000           # full-batch L-BFGS polish after Adam (0 to disable)

# # --- residual-based adaptive refinement (RAR) of NS collocation points in the gap ---
# RAR_EPOCHS = [EPOCHS_ADAM // 4, EPOCHS_ADAM // 2, (3 * EPOCHS_ADAM) // 4]
# RAR_CANDIDATES, RAR_ADD = 20000, 500

# # =====================================================================
# # SETUP
# # =====================================================================
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# os.makedirs(SAVE_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # ------------------ LOAD + CROP ------------------
# data = np.load(DATA_PATH)

# u_grid = data["u"]
# v_grid = data["v"]

# ny, nx = u_grid.shape

# crop_y, crop_x = 100, 100

# y_start = max((ny - crop_y) // 2, 0)
# y_end   = min(y_start + crop_y, ny)

# x_start = 0
# x_end   = min(crop_x, nx)

# u_grid = u_grid[y_start:y_end, x_start:x_end]
# v_grid = v_grid[y_start:y_end, x_start:x_end]

# ny, nx = u_grid.shape

# x_axis = np.arange(nx)
# y_axis = np.arange(ny)
# X, Y = np.meshgrid(x_axis, y_axis)

# print(f"Cropped shape: ({ny}, {nx})")

# X_MIN, X_MAX = x_axis.min(), x_axis.max()
# Y_MIN, Y_MAX = y_axis.min(), y_axis.max()

# # ------------------ GEOMETRY ------------------
# cx, cy = 50, 50
# r = 25
# assert abs(2 * r - D_idx) < 1e-9, "D_idx must equal 2r"

# dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
# cylinder = dist <= r

# R_outer = r + gap_thickness
# gap_mask = (dist <= R_outer) & (~cylinder)          # evaluation gap (from the geometric surface)

# GD = gap_thickness / (2 * r)
# RUN_TAG = f"gD{GD:.3f}"          # post-processing script must use the same tag
# print("Run tag:", RUN_TAG)

# # ------------------ NaN CLEANING ------------------
# u_grid = np.where(cylinder, 0.0, u_grid)
# v_grid = np.where(cylinder, 0.0, v_grid)

# ok = np.isfinite(u_grid) & np.isfinite(v_grid)
# print(f"NaN nodes outside cylinder: {(~ok).sum()} (of which in gap: {((~ok) & gap_mask).sum()})")

# _, (_iy, _ix) = distance_transform_edt(~(ok & ~cylinder), return_indices=True)
# u_grid = np.where(ok, u_grid, u_grid[_iy, _ix])
# v_grid = np.where(ok, v_grid, v_grid[_iy, _ix])

# fluid_eval = (~cylinder) & ok
# gap_eval   = gap_mask & ok


# # ------------------ EFFECTIVE NO-SLIP RADIUS ------------------
# def calibrate_wall_offset():
#     """Immersed-boundary data has a smeared, ~stagnant band next to the geometric surface.
#     Extrapolate the near-wall tangential velocity linearly to zero over several windows and angles;
#     n0 is the wall distance [cells] where u_t would vanish.
#     NOTE: this is a one-off calibration of the REFERENCE solver's effective wall using gap data.
#     For experimental data the wall position comes from the known geometry instead."""
#     th_all = np.arctan2(Y - cy, X - cx)
#     n_all = dist - r
#     side = np.where(Y - cy > 0, 1.0, -1.0)
#     ut = side * (u_grid * np.sin(th_all) - v_grid * np.cos(th_all))   # flow-aligned tangential velocity
#     phi = np.degrees(np.pi - np.abs(th_all))                          # angle from the front stagnation point
#     out = {}
#     print("Wall-offset calibration (linear extrapolation of u_t to zero):")
#     for lo, hi in ((1.5, 3.5), (2.0, 4.0), (3.0, 5.0)):
#         n0s = []
#         for p0 in range(30, 100, 10):
#             m = (n_all >= lo) & (n_all <= hi) & (phi >= p0) & (phi < p0 + 10) & ok & (~cylinder)
#             if m.sum() < 5:
#                 continue
#             a, b = np.polyfit(n_all[m], ut[m], 1)
#             if a > 0:
#                 n0s.append(-b / a)
#         out[(lo, hi)] = float(np.median(n0s)) if n0s else float("nan")
#         print(f"  n in [{lo}, {hi}] cells: n0 = {out[(lo, hi)]:.2f}  ({len(n0s)} angle bins)")
#     return out[(2.0, 4.0)]


# if WALL_OFFSET is None:
#     _n0 = calibrate_wall_offset()
#     WALL_OFFSET = float(np.clip(np.nan_to_num(_n0), 0.0, max(gap_thickness - 1.0, 0.0)))
# R_WALL = r + WALL_OFFSET
# D_LEN = max(R_outer - R_WALL, 1.0)           # effective fluid-gap thickness [cells]
# print(f"Effective no-slip radius: r_wall = {R_WALL:.2f} (offset {WALL_OFFSET:.2f} cells), fluid gap = {D_LEN:.2f} cells")

# # ------------------ TENSORS ------------------
# def to_t(arr, grad_=False):
#     t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
#     return t.requires_grad_(True) if grad_ else t

# # GAP interior (outside the effective wall) -> NS only (no velocity labels used)
# gap_ns_mask = (dist > R_WALL) & (dist <= R_outer)
# X_gap_t = to_t(X[gap_ns_mask], True)
# Y_gap_t = to_t(Y[gap_ns_mask], True)

# # Labelled ring outside the gap -> data anchor (thick: gives value and slope information at the interface)
# ring_mask = (dist > R_outer) & (dist <= R_outer + RING_WIDTH) & ok

# X_b_t = to_t(X[ring_mask])
# Y_b_t = to_t(Y[ring_mask])
# U_b_t = to_t(u_grid[ring_mask])
# V_b_t = to_t(v_grid[ring_mask])

# # DATA outside gap and ring
# data_mask = (~cylinder) & (~gap_mask) & (~ring_mask) & ok

# X_out_t = to_t(X[data_mask])
# Y_out_t = to_t(Y[data_mask])
# U_out_t = to_t(u_grid[data_mask])
# V_out_t = to_t(v_grid[data_mask])

# print(f"GAP interior (NS): {len(X_gap_t):,}")
# print(f"RING (labelled):   {len(X_b_t):,}")
# print(f"DATA (outside):    {len(X_out_t):,}")

# # Wall-biased NS collocation points between the effective wall and the gap edge (RAR appends to this set)
# th_w = 2 * np.pi * np.random.rand(N_wall)
# rr_w = R_WALL + (R_outer - R_WALL) * np.random.rand(N_wall) ** 2
# X_wall_t = to_t(cx + rr_w * np.cos(th_w), True)
# Y_wall_t = to_t(cy + rr_w * np.sin(th_w), True)

# # ------------------ MODEL ------------------
# def normalise(val, vmin, vmax):
#     return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


# class PINN(nn.Module):
#     """Inputs: normalised (x, y) plus a wall-distance feature. Outputs (u, v, p) in data units.
#     Hard no-slip: (u, v) = tanh(d/l) * net_uv for d >= 0 and exactly 0 inside the effective wall."""
#     def __init__(self):
#         super().__init__()
#         layers = [nn.Linear(3, H), nn.Tanh()]
#         for _ in range(N_HIDDEN - 1):
#             layers += [nn.Linear(H, H), nn.Tanh()]
#         layers += [nn.Linear(H, 3)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, x, y):
#         xn = normalise(x, X_MIN, X_MAX)
#         yn = normalise(y, Y_MIN, Y_MAX)
#         rho = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12)
#         d = rho - R_WALL                                   # signed wall distance [cells]
#         feat = torch.tanh(d / (3.0 * D_LEN))               # resolves the near-wall layer, saturates far away
#         out = self.net(torch.cat([xn, yn, feat], dim=1))
#         u, v, p = out.split(1, dim=1)
#         if HARD_NOSLIP:
#             phi = torch.tanh(d / D_LEN) * (d >= 0).to(out.dtype)   # one-sided: phi'(0+) = 1/D_LEN
#             u, v = phi * u, phi * v
#         return torch.cat([u, v, p], dim=1)


# def grad(f, x):
#     return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


# model = PINN().to(device)
# params = list(model.parameters())
# print(f"Parameters: {sum(p.numel() for p in params):,}")

# # ------------------ LOSSES ------------------
# FULL_BATCH = False          # switched on for L-BFGS (deterministic loss)


# def pick(n_total, b):
#     if FULL_BATCH or b >= n_total:
#         return torch.arange(n_total, device=device)
#     return torch.randperm(n_total, device=device)[:b]


# def ns_point_res(x, y):
#     u, v, p = model(x, y).split(1, dim=1)

#     ux, uy = grad(u, x), grad(u, y)
#     vx, vy = grad(v, x), grad(v, y)

#     uxx, uyy = grad(ux, x), grad(uy, y)
#     vxx, vyy = grad(vx, x), grad(vy, y)

#     px, py = grad(p, x), grad(p, y)

#     cont = ux + vy
#     momx = u*ux + v*uy - nu*(uxx + uyy) + px
#     momy = u*vx + v*vy - nu*(vxx + vyy) + py
#     return cont**2 + momx**2 + momy**2


# def NS_loss_fn(x, y):
#     return torch.mean(ns_point_res(x, y))


# def NS_loss():
#     i1 = pick(len(X_gap_t), b_ns // 2)
#     i2 = pick(len(X_wall_t), b_ns // 2)
#     return NS_loss_fn(X_gap_t[i1], Y_gap_t[i1]) + 5 * NS_loss_fn(X_wall_t[i2], Y_wall_t[i2])


# def data_loss():
#     idx = pick(len(X_out_t), b_data)
#     u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)
#     return torch.mean((u_p - U_out_t[idx])**2) + torch.mean((v_p - V_out_t[idx])**2)


# _a_fix = torch.linspace(0, 2 * np.pi, b_bc, device=device).unsqueeze(1)


# def bc_wall():
#     """Soft no-slip on the effective wall circle (only used when HARD_NOSLIP is False)."""
#     a = _a_fix if FULL_BATCH else 2 * np.pi * torch.rand(b_bc, 1, device=device)
#     x = cx + R_WALL * torch.cos(a)
#     y = cy + R_WALL * torch.sin(a)
#     u, v, _ = model(x, y).split(1, dim=1)
#     return torch.mean(u**2 + v**2)


# def bc_gap():
#     idx = pick(len(X_b_t), b_gap)
#     u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)
#     return torch.mean((u_p - U_b_t[idx])**2) + torch.mean((v_p - V_b_t[idx])**2)


# w = dict(W_INIT)


# def loss_terms():
#     ns = NS_loss()
#     dat = data_loss()
#     gp = bc_gap()
#     bc = torch.zeros((), device=device) if HARD_NOSLIP else bc_wall()
#     return ns, dat, bc, gp


# def total_loss(ns, dat, bc, gp):
#     L = w["ns"] * ns + w["data"] * dat + w["gap"] * gp
#     if not HARD_NOSLIP:
#         L = L + w["bc"] * bc
#     return L


# # ------------------ ADAPTIVE LOSS WEIGHTS (gradient-norm balancing) ------------------
# def grad_norm(term):
#     g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
#     return torch.sqrt(sum((gi ** 2).sum() for gi in g if gi is not None) + 1e-30)


# def update_weights(terms):
#     ref = w["ns"] * grad_norm(terms["ns"]).item()
#     for k, t in terms.items():
#         if k == "ns":
#             continue
#         lam_hat = ref / (grad_norm(t).item() + 1e-12)
#         w[k] = float(np.clip(ADAPT_ALPHA * w[k] + (1 - ADAPT_ALPHA) * lam_hat, 1e-3, 1e9))


# # ------------------ RESIDUAL-BASED ADAPTIVE REFINEMENT ------------------
# def rar_step():
#     global X_wall_t, Y_wall_t
#     th = 2 * np.pi * np.random.rand(RAR_CANDIDATES)
#     rr = np.sqrt(R_WALL ** 2 + np.random.rand(RAR_CANDIDATES) * (R_outer ** 2 - R_WALL ** 2))   # area-uniform
#     xc = to_t(cx + rr * np.cos(th), True)
#     yc = to_t(cy + rr * np.sin(th), True)
#     res = ns_point_res(xc, yc).detach().squeeze(1)
#     top = torch.topk(res, RAR_ADD).indices
#     X_wall_t = torch.cat([X_wall_t.detach(), xc.detach()[top]]).requires_grad_(True)
#     Y_wall_t = torch.cat([Y_wall_t.detach(), yc.detach()[top]]).requires_grad_(True)
#     print(f"  RAR: +{RAR_ADD} points, max residual {res.max().item():.3e}, NS set size {len(X_wall_t):,}")


# # ------------------ PHYSICS-ONLY DIAGNOSTIC (no ground truth) ------------------
# _gap_eval_idx = np.random.RandomState(SEED).choice(
#     len(X_gap_t), size=min(1000, len(X_gap_t)), replace=False)
# _Xe = X_gap_t.detach()[_gap_eval_idx].clone()
# _Ye = Y_gap_t.detach()[_gap_eval_idx].clone()


# def gap_residual():
#     x = _Xe.clone().requires_grad_(True)
#     y = _Ye.clone().requires_grad_(True)
#     return NS_loss_fn(x, y).item()


# # ------------------ HISTORY ------------------
# history = {k: [] for k in ["epoch", "loss_total", "loss_ns", "loss_data", "loss_bc",
#                            "loss_bc_gap", "gap_resid", "rel_l2_full", "rel_l2_gap"]}

# X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
# Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

# U_true = np.sqrt(u_grid**2 + v_grid**2)
# gap_eval_w = gap_eval & (dist > R_WALL)         # gap measured from the effective wall


# def log_history(epoch, L_val, ns_v, dat_v, bc_v, gp_v):
#     with torch.no_grad():
#         u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
#         U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()
#     rel_full = np.linalg.norm((U_pred - U_true)[fluid_eval]) / np.linalg.norm(U_true[fluid_eval])
#     rel_gap = np.linalg.norm((U_pred - U_true)[gap_eval]) / np.linalg.norm(U_true[gap_eval])
#     history["epoch"].append(epoch)
#     history["loss_total"].append(L_val)
#     history["loss_ns"].append(ns_v)
#     history["loss_data"].append(dat_v)
#     history["loss_bc"].append(bc_v)
#     history["loss_bc_gap"].append(gp_v)
#     history["gap_resid"].append(gap_residual())
#     history["rel_l2_full"].append(rel_full)
#     history["rel_l2_gap"].append(rel_gap)
#     return U_pred


# # =====================================================================
# # TRAINING: Adam (+ adaptive weights + RAR), then full-batch L-BFGS
# # =====================================================================
# optimizer = torch.optim.Adam(params, lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(EPOCHS_ADAM // 3, 1), gamma=0.5)

# for epoch in range(EPOCHS_ADAM):
#     optimizer.zero_grad()
#     ns, dat, bc, gp = loss_terms()

#     if ADAPTIVE and epoch % ADAPT_EVERY == 0:
#         terms = {"ns": ns, "data": dat, "gap": gp}
#         if not HARD_NOSLIP:
#             terms["bc"] = bc
#         update_weights(terms)

#     L = total_loss(ns, dat, bc, gp)
#     L.backward()
#     optimizer.step()
#     scheduler.step()

#     if epoch in RAR_EPOCHS:
#         rar_step()

#     if epoch % 50 == 0:
#         log_history(epoch, L.item(), ns.item(), dat.item(), bc.item(), gp.item())

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch:5d} | Loss {history['loss_total'][-1]:.3e} | "
#               f"NS {history['loss_ns'][-1]:.3e} | Data {history['loss_data'][-1]:.3e} | "
#               f"BC {history['loss_bc'][-1]:.3e} | Ring {history['loss_bc_gap'][-1]:.3e} | "
#               f"GapRes {history['gap_resid'][-1]:.3e} | L2 gap {history['rel_l2_gap'][-1]:.4e} | "
#               f"w {{{', '.join(f'{k}:{v:.3g}' for k, v in w.items())}}}")

# if LBFGS_ITERS > 0:
#     print("L-BFGS polish (full batch, weights frozen) ...")
#     FULL_BATCH = True
#     lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_ITERS, max_eval=int(1.25 * LBFGS_ITERS),
#                               history_size=50, tolerance_grad=1e-10, tolerance_change=1e-14,
#                               line_search_fn="strong_wolfe")

#     def closure():
#         lbfgs.zero_grad()
#         ns_, dat_, bc_, gp_ = loss_terms()
#         L_ = total_loss(ns_, dat_, bc_, gp_)
#         L_.backward()
#         return L_

#     lbfgs.step(closure)

#     ns, dat, bc, gp = loss_terms()
#     L = total_loss(ns, dat, bc, gp)
#     log_history(EPOCHS_ADAM + LBFGS_ITERS, L.item(), ns.item(), dat.item(), bc.item(), gp.item())
#     FULL_BATCH = False
#     print(f"After L-BFGS | Loss {history['loss_total'][-1]:.3e} | NS {history['loss_ns'][-1]:.3e} | "
#           f"L2 gap {history['rel_l2_gap'][-1]:.4e}")

# # Final summary (gap measured from the geometric surface AND from the effective wall)
# model.eval()
# with torch.no_grad():
#     u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
#     U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()
# print(f"Final rel L2 |U|: gap (from r) {np.linalg.norm((U_pred - U_true)[gap_eval]) / np.linalg.norm(U_true[gap_eval]):.4e} | "
#       f"gap (from r_wall) {np.linalg.norm((U_pred - U_true)[gap_eval_w]) / np.linalg.norm(U_true[gap_eval_w]):.4e} | "
#       f"fluid {np.linalg.norm((U_pred - U_true)[fluid_eval]) / np.linalg.norm(U_true[fluid_eval]):.4e}")

# # ------------------ SAVE MODEL + POST-PROCESSING DATA ------------------
# model_path = os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt")
# post_path  = os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz")

# torch.save(model.state_dict(), model_path)

# np.savez(
#     post_path,
#     X=X, Y=Y,
#     cylinder=cylinder, gap_mask=gap_mask, ok=ok,
#     cx=cx, cy=cy, r=r, gap_thickness=gap_thickness,
#     r_wall=R_WALL, d_len=D_LEN, hard_noslip=int(HARD_NOSLIP),
#     u_grid=u_grid, v_grid=v_grid,
#     h=H, n_hidden=N_HIDDEN, nu=nu,
#     **{f"hist_{k}": np.array(v) for k, v in history.items()},
# )

# print(f"Saved model -> {model_path}")
# print(f"Saved post-processing data -> {post_path}")