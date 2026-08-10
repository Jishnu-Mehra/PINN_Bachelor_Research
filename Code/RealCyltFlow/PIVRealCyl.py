import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------ LOAD PARTICLE TRACKS ------------------
file_path = "PINN_Bachelor_Research/Data/RealCylFlow/xyzuvw_tracks.npy"

tracks = np.load(file_path)  # shape (N, 6): x, y, z, u, v, w
print("Loaded shape:", tracks.shape)

x_all = tracks[:, 0]
y_all = tracks[:, 1]
z_all = tracks[:, 2]
u_all = tracks[:, 3]
v_all = tracks[:, 4]

# ------------------ FIND MID-SPAN Z ------------------
z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)
z_mid = 0.5 * (z_min + z_max)
print(f"z range: [{z_min:.3f}, {z_max:.3f}], taking mid-span z = {z_mid:.3f}")

z_tol = 2.0  # mm half-width of the slab; widen if too few points survive
z_mask = np.abs(z_all - z_mid) <= z_tol

x_full = x_all[z_mask]
y_full = y_all[z_mask]
u_full = u_all[z_mask]
v_full = v_all[z_mask]

print(f"Points in mid-span slice: {x_full.shape[0]} / {x_all.shape[0]}")

# NaN filter
valid = ~np.isnan(x_full) & ~np.isnan(y_full) & ~np.isnan(u_full) & ~np.isnan(v_full)
x_full, y_full, u_full, v_full = x_full[valid], y_full[valid], u_full[valid], v_full[valid]
print(f"Valid points after NaN filter: {x_full.shape[0]}")

# mm/s -> m/s (drop if your tracks are already SI)
u_full = u_full / 1000.0
v_full = v_full / 1000.0

# ------------------ PHYSICAL CROPPING ------------------
x_min = np.min(x_full)
x_max_crop = 300

y_min_crop = -100
y_max_crop = 100

crop_mask = (
    (x_full >= x_min) & (x_full <= x_max_crop) &
    (y_full >= y_min_crop) & (y_full <= y_max_crop)
)

x = x_full[crop_mask]
y = y_full[crop_mask]
u_grid = u_full[crop_mask]   # kept name u_grid/v_grid so downstream refs still match
v_grid = v_full[crop_mask]

print(f"Cropped point count: {x.shape[0]}")

# bounds (needed for model input normalisation)
X_MIN, X_MAX = x.min(), x.max()
Y_MIN, Y_MAX = y.min(), y.max()

# ------------------ GEOMETRY ------------------
cx, cy = 0, 0
d = 125
r = d / 2
ratio = 0.04
gap_thickness = ratio * d

dist2 = (x - cx)**2 + (y - cy)**2
cylinder = dist2 <= r**2

r_inner = r
r_outer = r + gap_thickness

gap_mask = (dist2 >= r_inner**2) & (dist2 <= r_outer**2)
gap_mask = gap_mask & (~cylinder)

# boundary: thin annulus just outside the gap ring
tol = (X_MAX - X_MIN) / 500
dist = np.sqrt(dist2)

boundary_mask = (np.abs(dist - r_outer) < tol)
boundary_mask = boundary_mask & (~cylinder) & (~gap_mask)

# data outside
data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask)

print(f"cylinder: {cylinder.sum()}, gap: {gap_mask.sum()}, "
      f"boundary: {boundary_mask.sum()}, data(outer): {data_mask.sum()}")

# NOTE: with scattered points there's no grid-wide NaN mask like before —
# `valid` was already applied above during loading, so every point in
# x, y, u_grid, v_grid is already finite. No separate `valid_mask` needed
# downstream (the old grid version needed one because NaNs sat inside the
# array at fixed grid positions; here invalid rows were simply dropped).

# ------------------ TENSORS ------------------
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)

X_gap_t = to_tensor(x[gap_mask]).requires_grad_(True)
Y_gap_t = to_tensor(y[gap_mask]).requires_grad_(True)

U_gap_t = to_tensor(u_grid[gap_mask])
V_gap_t = to_tensor(v_grid[gap_mask])

X_b_t = to_tensor(x[boundary_mask])
Y_b_t = to_tensor(y[boundary_mask])

U_b_t = to_tensor(u_grid[boundary_mask])
V_b_t = to_tensor(v_grid[boundary_mask])

X_out_t = to_tensor(x[data_mask])
Y_out_t = to_tensor(y[data_mask])

U_out_t = to_tensor(u_grid[data_mask])
V_out_t = to_tensor(v_grid[data_mask])

print(f"GAP interior: {len(X_gap_t)}")
print(f"GAP boundary: {len(X_b_t)}")
print(f"DATA: {len(X_out_t)}")

N_wall = 30000

theta_wall = 2*np.pi*np.random.rand(N_wall)
rr = r + gap_thickness * (np.random.rand(N_wall)**2)

x_wall_ns = cx + rr*np.cos(theta_wall)
y_wall_ns = cy + rr*np.sin(theta_wall)

X_wall_ns_t = torch.tensor(
    x_wall_ns,
    dtype=torch.float32,
    device=device
).unsqueeze(1).requires_grad_(True)

Y_wall_ns_t = torch.tensor(
    y_wall_ns,
    dtype=torch.float32,
    device=device
).unsqueeze(1).requires_grad_(True)

# ------------------ HYPERPARAMS ------------------
h = 64
nu = 1.5e-5

b_ns   = 2000
b_data = 10000
b_bc   = 5000
b_gap  = 1000

w_ns   = 0.1
w_bc   = 500
w_data = 100
w_gap  = 50

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
            nn.Linear(h, 3)
        )

    def forward(self, x, y):
        xn = normalise(x, X_MIN, X_MAX)
        yn = normalise(y, Y_MIN, Y_MAX)
        return self.net(torch.cat([xn, yn], dim=1))


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

    ux = grad(u, x);  uy = grad(u, y)
    vx = grad(v, x);  vy = grad(v, y)

    uxx = grad(ux, x); uyy = grad(uy, y)
    vxx = grad(vx, x); vyy = grad(vy, y)

    px = grad(p, x);  py = grad(p, y)

    cont = ux + vy
    momx = u*ux + v*uy - nu*(uxx + uyy) + px
    momy = u*vx + v*vy - nu*(vxx + vyy) + py

    return (
        torch.mean(cont**2) +
        torch.mean(momx**2) +
        torch.mean(momy**2)
    )

def NS_loss():
    idx1 = torch.randperm(len(X_gap_t),     device=device)[:b_ns//2]
    idx2 = torch.randperm(len(X_wall_ns_t), device=device)[:b_ns//2]

    loss_gap  = NS_loss_fn(X_gap_t[idx1],     Y_gap_t[idx1])
    loss_wall = NS_loss_fn(X_wall_ns_t[idx2], Y_wall_ns_t[idx2])

    return loss_gap + 5*loss_wall

def data_loss():
    if w_data == 0:
        return torch.tensor(0.0, device=device)

    idx = torch.randperm(len(X_out_t), device=device)[:b_data]
    u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)

    return (
        torch.mean((u_p - U_out_t[idx])**2) +
        torch.mean((v_p - V_out_t[idx])**2)
    )

def bc_cylinder():
    a = 2*np.pi*torch.rand(b_bc, 1, device=device)
    xc = cx + r*torch.cos(a)
    yc = cy + r*torch.sin(a)
    u, v, _ = model(xc, yc).split(1, dim=1)
    return torch.mean(u**2 + v**2)

def bc_gap():
    idx = torch.randperm(len(X_b_t), device=device)[:b_gap]
    u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)
    return (
        torch.mean((u_p - U_b_t[idx])**2) +
        torch.mean((v_p - V_b_t[idx])**2)
    )

def loss():
    ns  = NS_loss()
    dat = data_loss()
    bc  = bc_cylinder()
    gap = bc_gap()

    total = (
        w_ns*ns +
        w_data*dat +
        w_bc*bc +
        w_gap*gap
    )

    return total, ns.item(), dat.item(), bc.item(), gap.item()

# ------------------ TRAINING + HISTORY ------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

checkpoints = [0, 500, 1000, 2000]

history = {
    "epoch":       [],
    "loss_total":  [],
    "loss_ns":     [],
    "loss_data":   [],
    "loss_bc":     [],
    "loss_bc_gap": [],
    "rel_l2_full": [],
    "rel_l2_gap":  [],
}

# NOTE: previously X_flat_t/Y_flat_t were the flattened *grid* for dense
# evaluation + reshape-based plotting. With scattered data there's no dense
# grid to flatten — we evaluate the model directly at the data point
# locations (x, y) themselves, which are already flat 1D arrays.
X_flat_t = to_tensor(x)
Y_flat_t = to_tensor(y)

U_true = np.sqrt(u_grid**2 + v_grid**2)

for epoch in range(2001):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val, bcgap_val = loss()
    L.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # ----- Metrics -----
    if epoch % 50 == 0:
        with torch.no_grad():
            u_p, v_p, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u_p**2 + v_p**2).cpu().numpy().flatten()

        # no grid -> no valid_mask; every point here is already valid
        outside = ~cylinder

        rel_l2_full = np.linalg.norm(U_pred[outside] - U_true[outside]) / np.linalg.norm(U_true[outside])

        gap_valid  = gap_mask & outside
        rel_l2_gap = np.linalg.norm(U_pred[gap_valid] - U_true[gap_valid]) / np.linalg.norm(U_true[gap_valid])

        history["epoch"].append(epoch)
        history["loss_total"].append(L.item())
        history["loss_ns"].append(ns_val)
        history["loss_data"].append(dat_val)
        history["loss_bc"].append(bc_val)
        history["loss_bc_gap"].append(bcgap_val)
        history["rel_l2_full"].append(rel_l2_full)
        history["rel_l2_gap"].append(rel_l2_gap)

    # ----- Logging -----
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {history['loss_total'][-1]:.3e} | "
              f"NS {history['loss_ns'][-1]:.3e} | "
              f"Data {history['loss_data'][-1]:.3e} | "
              f"BC {history['loss_bc'][-1]:.3e} | "
              f"BC_gap {history['loss_bc_gap'][-1]:.3e} | "
              f"L2 gap {history['rel_l2_gap'][-1]:.4e}")

    # ----- Checkpoint plots -----
    if epoch in checkpoints:
        fig, axs = plt.subplots(1, 2, figsize=(14, 3))

        with torch.no_grad():
            u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u_flat**2 + v_flat**2).cpu().numpy().flatten()

        outside = ~cylinder
        vmin   = np.min(U_true[outside])
        vmax   = np.max(U_true[outside])
        levels = np.linspace(vmin, vmax, 20)

        # tricontourf works directly on scattered (x, y, value) triples —
        # no reshape/grid needed. Points inside the cylinder are dropped
        # from the triangulation rather than masked post-hoc.
        cf_true = axs[0].tricontourf(x[outside], y[outside], U_true[outside],
                                      levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
        axs[0].add_patch(plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--'))
        axs[0].add_patch(plt.Circle((cx, cy), r,               color='k', fill=False, linestyle='--'))
        axs[0].set_title(f"Ground Truth |U| (Epoch {epoch})")
        axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].axis("equal")
        plt.colorbar(cf_true, ax=axs[0])

        cf_pred = axs[1].tricontourf(x[outside], y[outside], U_pred[outside],
                                      levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
        axs[1].add_patch(plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--'))
        axs[1].add_patch(plt.Circle((cx, cy), r,               color='k', fill=False, linestyle='--'))
        axs[1].set_title(f"PINN Prediction |U| (Epoch {epoch})")
        axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].axis("equal")
        plt.colorbar(cf_pred, ax=axs[1])

        plt.suptitle(f"Epoch {epoch} Comparison", fontsize=14)
        plt.tight_layout()
        plt.show()

# ------------------ FINAL PLOTS ------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(history["epoch"], history["loss_ns"],     label="NS")
axs[0].plot(history["epoch"], history["loss_data"],   label="Data")
axs[0].plot(history["epoch"], history["loss_bc"],     label="BC (Cylinder)")
axs[0].plot(history["epoch"], history["loss_bc_gap"], label="BC (Gap)")
axs[0].set_yscale("log")
axs[0].set_xlabel("Epoch")
axs[0].set_title("Loss components (unweighted)")
axs[0].legend()
axs[0].grid(True, which="both", ls="--", alpha=0.5)

axs[1].plot(history["epoch"], history["rel_l2_gap"],  label="Gap only")
axs[1].plot(history["epoch"], history["rel_l2_full"], label="Full domain", linestyle="--")
axs[1].set_yscale("log")
axs[1].set_xlabel("Epoch")
axs[1].set_title("Relative $L_2$ Error")
axs[1].legend()
axs[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()

# ------------------ HYBRID PLOT ------------------
model.eval()

with torch.no_grad():
    u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
    u_pred = u_flat.cpu().numpy().flatten()
    v_pred = v_flat.cpu().numpy().flatten()
    U_pred = np.sqrt(u_pred**2 + v_pred**2)

U_true = np.sqrt(u_grid**2 + v_grid**2)
U_hybrid = U_true.copy()
U_hybrid[gap_mask] = U_pred[gap_mask]

outside = ~cylinder
vmin = np.min(U_true[outside])
vmax = np.max(U_true[outside])

# scattered plotting: drop cylinder points instead of masking a grid array
plt.tricontourf(x[outside], y[outside], U_hybrid[outside],
                 levels=20, cmap="jet", vmin=vmin, vmax=vmax)
plt.gca().set_aspect('equal')
plt.gca().add_patch(plt.Circle((cx, cy), r,               color='k', fill=False, linestyle='--'))
plt.gca().add_patch(plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--'))
plt.title("PINN in Gap + Ground Truth Outside")
plt.colorbar()
plt.show()

# ------------------ SAVE ------------------
# torch.save(model.state_dict(), "cylinder_pinn_imp0.04.pt")
print("Model saved.")

# Saved as flat point arrays (x, y, ...) instead of 2D grids (X, Y, ...).
# Anything reloading this file downstream (e.g. your postprocessing script)
# needs the same scatter-based treatment — no .reshape(X.shape) available.
# np.savez(
#     "postprocess_data_diag.npz",
#     x=x, y=y,
#     u_grid=u_grid, v_grid=v_grid,
#     u_pred=u_pred, v_pred=v_pred, U_mag=U_pred,
#     gap_mask=gap_mask,
#     cylinder=cylinder,
#     cx=cx, cy=cy, r=r,
#     gap_thickness=gap_thickness
# )