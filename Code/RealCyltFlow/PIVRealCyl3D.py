import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers '3d' projection
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==================================================================
# 3D extension of the cylinder-gap PINN.
#
# Key differences from the 2D (mid-span slice) version:
#   - Input is (x, y, z) instead of (x, y); output is (u, v, w, p)
#     instead of (u, v, p).
#   - The cylinder and gap are now 3D: a circular cross-section in the
#     x-y plane, extruded along the full z-span present in the data
#     (not just a single mid-span slice).
#   - The NS residual is the full 3D incompressible Navier-Stokes
#     equations (continuity + 3 momentum components), which needs 9
#     second derivatives per collocation point via autograd instead of
#     2D's 6 -- noticeably heavier per training step.
#   - Skin friction C_f is now a function of both azimuthal angle theta
#     AND spanwise position z, not just theta.
#
# This script keeps epochs/collocation counts modest by default so it
# runs as a first sanity check of the 3D mechanics. Scale up
# N_COLLOC_* / EPOCHS once you've confirmed it trains sensibly.
# ==================================================================

# ------------------ LOAD 3D PARTICLE TRACKS ------------------
file_path = "PINN_Bachelor_Research/Data/RealCylFlow/xyzuvw_tracks.npy"

tracks = np.load(file_path)  # shape (N, 6): x, y, z, u, v, w
print("Loaded shape:", tracks.shape)

x_all = tracks[:, 0]
y_all = tracks[:, 1]
z_all = tracks[:, 2]
u_all = tracks[:, 3]
v_all = tracks[:, 4]
w_all = tracks[:, 5]

# NaN filter (no mid-span slicing this time -- keep the full volume)
valid = (
    ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isnan(z_all)
    & ~np.isnan(u_all) & ~np.isnan(v_all) & ~np.isnan(w_all)
)
x_full, y_full, z_full = x_all[valid], y_all[valid], z_all[valid]
u_full, v_full, w_full = u_all[valid], v_all[valid], w_all[valid]
print(f"Valid points after NaN filter: {x_full.shape[0]} / {x_all.shape[0]}")

# mm/s -> m/s (drop if your tracks are already SI -- same convention as
# the 2D script)
u_full = u_full / 1000.0
v_full = v_full / 1000.0
w_full = w_full / 1000.0

# ------------------ PHYSICAL CROPPING ------------------
# Same x/y crop as the 2D version. z is NOT cropped to a mid-span slice
# here -- we keep the full spanwise extent present in the data. If your
# rig has strong end effects near the tunnel walls, you may want to
# crop z too; left uncropped for now so you can see the raw spanwise
# behaviour first.
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
z = z_full[crop_mask]
u_grid = u_full[crop_mask]
v_grid = v_full[crop_mask]
w_grid = w_full[crop_mask]

print(f"Cropped point count: {x.shape[0]}")
print(f"z range in cropped data: [{z.min():.2f}, {z.max():.2f}]")

# bounds (needed for model input normalisation)
X_MIN, X_MAX = x.min(), x.max()
Y_MIN, Y_MAX = y.min(), y.max()
Z_MIN, Z_MAX = z.min(), z.max()

# ------------------ GEOMETRY (3D: cylinder extruded along z) ------------------
cx, cy = 0, 0
d = 125
r = d / 2
ratio = 0.04
gap_thickness = ratio * d

# distance is now purely in the x-y cross-section (cylinder axis is z)
dist2 = (x - cx)**2 + (y - cy)**2
cylinder = dist2 <= r**2

r_inner = r
r_outer = r + gap_thickness

gap_mask = (dist2 >= r_inner**2) & (dist2 <= r_outer**2)
gap_mask = gap_mask & (~cylinder)

# boundary: thin annular SHELL just outside the gap ring, extruded
# along the full z-span (not a single ring like the 2D case)
tol = (X_MAX - X_MIN) / 500
dist = np.sqrt(dist2)

boundary_mask = (np.abs(dist - r_outer) < tol)
boundary_mask = boundary_mask & (~cylinder) & (~gap_mask)

# data outside
data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask)

print(f"cylinder: {cylinder.sum()}, gap: {gap_mask.sum()}, "
      f"boundary: {boundary_mask.sum()}, data(outer): {data_mask.sum()}")

# ------------------ TENSORS ------------------
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)

X_gap_t = to_tensor(x[gap_mask]).requires_grad_(True)
Y_gap_t = to_tensor(y[gap_mask]).requires_grad_(True)
Z_gap_t = to_tensor(z[gap_mask]).requires_grad_(True)

X_b_t = to_tensor(x[boundary_mask])
Y_b_t = to_tensor(y[boundary_mask])
Z_b_t = to_tensor(z[boundary_mask])

U_b_t = to_tensor(u_grid[boundary_mask])
V_b_t = to_tensor(v_grid[boundary_mask])
W_b_t = to_tensor(w_grid[boundary_mask])

X_out_t = to_tensor(x[data_mask])
Y_out_t = to_tensor(y[data_mask])
Z_out_t = to_tensor(z[data_mask])

U_out_t = to_tensor(u_grid[data_mask])
V_out_t = to_tensor(v_grid[data_mask])
W_out_t = to_tensor(w_grid[data_mask])

print(f"GAP interior: {len(X_gap_t)}")
print(f"GAP boundary: {len(X_b_t)}")
print(f"DATA: {len(X_out_t)}")

# ------------------ NS COLLOCATION POINTS (3D annular shell + wall) ------------------
# Sampled throughout the annular shell (r to r_outer) in x-y, AND across
# the full z-span, biased toward the wall the same way the 2D script
# biases toward r (rand**2 skews samples toward r_inner).
N_wall = 20000  # reduced from 2D's 30000 -- 3D autograd is heavier per point

theta_wall = 2 * np.pi * np.random.rand(N_wall)
rr = r + gap_thickness * (np.random.rand(N_wall) ** 2)
zz = Z_MIN + (Z_MAX - Z_MIN) * np.random.rand(N_wall)

x_wall_ns = cx + rr * np.cos(theta_wall)
y_wall_ns = cy + rr * np.sin(theta_wall)
z_wall_ns = zz

X_wall_ns_t = to_tensor(x_wall_ns).requires_grad_(True)
Y_wall_ns_t = to_tensor(y_wall_ns).requires_grad_(True)
Z_wall_ns_t = to_tensor(z_wall_ns).requires_grad_(True)

# ------------------ HYPERPARAMS ------------------
h = 64
nu = 1.5e-5

b_ns   = 1500   # reduced from 2D's 2000 -- 9 second derivatives/pt is expensive
b_data = 8000   # reduced from 2D's 10000
b_bc   = 4000
b_gap  = 1000

w_ns   = 1
w_bc   = 500
w_data = 100
w_gap  = 50

EPOCHS = 1500  # reduced from 2D's 2001 -- this is a first sanity-check pass;
               # scale up once you've confirmed the loss curves behave

# ------------------ MODEL (3D input, 4D output) ------------------

def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


class PINN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, h), nn.Tanh(),   # 3 inputs: x, y, z
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 4)                # 4 outputs: u, v, w, p
        )

    def forward(self, x, y, z):
        xn = normalise(x, X_MIN, X_MAX)
        yn = normalise(y, Y_MIN, Y_MAX)
        zn = normalise(z, Z_MIN, Z_MAX)
        return self.net(torch.cat([xn, yn, zn], dim=1))


def grad(f, x):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]


model = PINN3D().to(device)

# ------------------ LOSSES (full 3D incompressible NS) ------------------

def NS_loss_fn(x, y, z):
    u, v, w, p = model(x, y, z).split(1, dim=1)

    ux = grad(u, x); uy = grad(u, y); uz = grad(u, z)
    vx = grad(v, x); vy = grad(v, y); vz = grad(v, z)
    wx = grad(w, x); wy = grad(w, y); wz = grad(w, z)

    uxx = grad(ux, x); uyy = grad(uy, y); uzz = grad(uz, z)
    vxx = grad(vx, x); vyy = grad(vy, y); vzz = grad(vz, z)
    wxx = grad(wx, x); wyy = grad(wy, y); wzz = grad(wz, z)

    px = grad(p, x); py = grad(p, y); pz = grad(p, z)

    cont = ux + vy + wz
    momx = u*ux + v*uy + w*uz - nu*(uxx + uyy + uzz) + px
    momy = u*vx + v*vy + w*vz - nu*(vxx + vyy + vzz) + py
    momz = u*wx + v*wy + w*wz - nu*(wxx + wyy + wzz) + pz

    return (
        torch.mean(cont**2) +
        torch.mean(momx**2) +
        torch.mean(momy**2) +
        torch.mean(momz**2)
    )

def NS_loss():
    idx1 = torch.randperm(len(X_gap_t),     device=device)[:b_ns//2]
    idx2 = torch.randperm(len(X_wall_ns_t), device=device)[:b_ns//2]

    loss_gap  = NS_loss_fn(X_gap_t[idx1], Y_gap_t[idx1], Z_gap_t[idx1])
    loss_wall = NS_loss_fn(X_wall_ns_t[idx2], Y_wall_ns_t[idx2], Z_wall_ns_t[idx2])

    return loss_gap + 5*loss_wall

def data_loss():
    idx = torch.randperm(len(X_out_t), device=device)[:b_data]
    u_p, v_p, w_p, _ = model(X_out_t[idx], Y_out_t[idx], Z_out_t[idx]).split(1, dim=1)

    return (
        torch.mean((u_p - U_out_t[idx])**2) +
        torch.mean((v_p - V_out_t[idx])**2) +
        torch.mean((w_p - W_out_t[idx])**2)
    )

def bc_cylinder():
    # sample points on the 3D cylindrical WALL surface: circle in x-y,
    # random z across the span
    a = 2 * np.pi * torch.rand(b_bc, 1, device=device)
    zc = Z_MIN + (Z_MAX - Z_MIN) * torch.rand(b_bc, 1, device=device)
    xc = cx + r * torch.cos(a)
    yc = cy + r * torch.sin(a)
    u, v, w, _ = model(xc, yc, zc).split(1, dim=1)
    return torch.mean(u**2 + v**2 + w**2)

def bc_gap():
    idx = torch.randperm(len(X_b_t), device=device)[:b_gap]
    u_p, v_p, w_p, _ = model(X_b_t[idx], Y_b_t[idx], Z_b_t[idx]).split(1, dim=1)
    return (
        torch.mean((u_p - U_b_t[idx])**2) +
        torch.mean((v_p - V_b_t[idx])**2) +
        torch.mean((w_p - W_b_t[idx])**2)
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.5)

checkpoints = [0, 500, 1000, EPOCHS - 1]

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

X_flat_t = to_tensor(x)
Y_flat_t = to_tensor(y)
Z_flat_t = to_tensor(z)

U_true = np.sqrt(u_grid**2 + v_grid**2 + w_grid**2)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val, bcgap_val = loss()
    L.backward()
    optimizer.step()
    scheduler.step()

    # ----- Metrics -----
    if epoch % 50 == 0:
        with torch.no_grad():
            u_p, v_p, w_p, _ = model(X_flat_t, Y_flat_t, Z_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u_p**2 + v_p**2 + w_p**2).cpu().numpy().flatten()

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

    # ----- Checkpoint plots: real 3D scatter comparison -----
    if epoch in checkpoints:
        with torch.no_grad():
            u_flat, v_flat, w_flat, _ = model(X_flat_t, Y_flat_t, Z_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u_flat**2 + v_flat**2 + w_flat**2).cpu().numpy().flatten()

        outside = ~cylinder
        vmin = np.min(U_true[outside])
        vmax = np.max(U_true[outside])

        # Subsample for plotting -- a full-resolution 3D scatter of every
        # point is unreadable and slow to render; a few thousand points
        # is enough to see the spatial pattern.
        N_PLOT = min(6000, outside.sum())
        plot_idx = np.random.choice(np.where(outside)[0], size=N_PLOT, replace=False)

        fig = plt.figure(figsize=(15, 6))

        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        sc0 = ax0.scatter(
            x[plot_idx], y[plot_idx], z[plot_idx],
            c=U_true[plot_idx], cmap="jet", vmin=vmin, vmax=vmax,
            s=4, alpha=0.6
        )
        ax0.set_title(f"Ground Truth |U| (Epoch {epoch})")
        ax0.set_xlabel("x"); ax0.set_ylabel("y"); ax0.set_zlabel("z")
        fig.colorbar(sc0, ax=ax0, shrink=0.6, label="|U|")

        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        sc1 = ax1.scatter(
            x[plot_idx], y[plot_idx], z[plot_idx],
            c=U_pred[plot_idx], cmap="jet", vmin=vmin, vmax=vmax,
            s=4, alpha=0.6
        )
        ax1.set_title(f"PINN Prediction |U| (Epoch {epoch})")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        fig.colorbar(sc1, ax=ax1, shrink=0.6, label="|U|")

        plt.suptitle(f"Epoch {epoch} Comparison (3D)", fontsize=14)
        plt.tight_layout()
        plt.show()

# ------------------ FINAL PLOTS ------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(history["epoch"], history["loss_ns"],     label="NS (3D)")
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
axs[1].set_title("Relative $L_2$ Error (3D |U|)")
axs[1].legend()
axs[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()

# ------------------ HYBRID PLOT (full 3D scatter) ------------------
model.eval()

with torch.no_grad():
    u_flat, v_flat, w_flat, _ = model(X_flat_t, Y_flat_t, Z_flat_t).split(1, dim=1)
    u_pred = u_flat.cpu().numpy().flatten()
    v_pred = v_flat.cpu().numpy().flatten()
    w_pred = w_flat.cpu().numpy().flatten()
    U_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

U_true = np.sqrt(u_grid**2 + v_grid**2 + w_grid**2)
U_hybrid = U_true.copy()
U_hybrid[gap_mask] = U_pred[gap_mask]

outside = ~cylinder
vmin = np.min(U_true[outside])
vmax = np.max(U_true[outside])

N_PLOT_FINAL = min(10000, outside.sum())
plot_idx = np.random.choice(np.where(outside)[0], size=N_PLOT_FINAL, replace=False)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    x[plot_idx], y[plot_idx], z[plot_idx],
    c=U_hybrid[plot_idx], cmap="jet", vmin=vmin, vmax=vmax,
    s=4, alpha=0.6
)

# Wireframe cylinder surface (inner wall) and outer gap boundary,
# extruded along the full z-span -- the 3D equivalent of the 2D
# script's dashed circles.
theta_cyl = np.linspace(0, 2*np.pi, 60)
z_cyl = np.linspace(Z_MIN, Z_MAX, 20)
theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)

x_cyl = cx + r * np.cos(theta_grid)
y_cyl = cy + r * np.sin(theta_grid)
ax.plot_wireframe(x_cyl, y_cyl, z_grid, color='k', linewidth=0.4, alpha=0.5)

x_gap_outer = cx + (r + gap_thickness) * np.cos(theta_grid)
y_gap_outer = cy + (r + gap_thickness) * np.sin(theta_grid)
ax.plot_wireframe(x_gap_outer, y_gap_outer, z_grid, color='gray', linewidth=0.3, linestyle='--', alpha=0.4)

ax.set_title("PINN in Gap + Ground Truth Outside (3D)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
fig.colorbar(sc, ax=ax, shrink=0.6, label="|U|")
plt.tight_layout()
plt.show()

# ------------------ SAVE ------------------
# torch.save(model.state_dict(), "cylinder_pinn_3d.pt")
print("Model trained (3D). Save uncommented to persist state_dict.")

# np.savez(
#     "postprocess_data_3d.npz",
#     x=x, y=y, z=z,
#     u_grid=u_grid, v_grid=v_grid, w_grid=w_grid,
#     u_pred=u_pred, v_pred=v_pred, w_pred=w_pred, U_mag=U_pred,
#     gap_mask=gap_mask,
#     cylinder=cylinder,
#     cx=cx, cy=cy, r=r,
#     gap_thickness=gap_thickness
# )