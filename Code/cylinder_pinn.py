# import numpy as np
# import scipy.io

# data = scipy.io.loadmat('PINN_Bachelor_Research/Data/Cylinder flow data/CYLINDER_ALL.mat')

# ny = data['nx'][0][0]
# nx = data['ny'][0][0]

# u_all = data['UALL']
# v_all = data['VALL']

# u_mean = np.mean(u_all, axis=1)
# v_mean = np.mean(v_all, axis=1)

# # reshape
# u_grid = u_mean.reshape((ny, nx), order='F')
# v_grid = v_mean.reshape((ny, nx), order='F')

# # save everything cleanly
# np.savez("cylinder_flow.npz",
#          u=u_grid,
#          v=v_grid,
#          nx=nx,
#          ny=ny)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ------------------ SETUP ------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------ LOAD + CROP ------------------
data = np.load("PINN_Bachelor_Research/Data/Cylinder flow data/cylinder_flow.npz")

u_grid = data["u"]
v_grid = data["v"]

ny, nx = u_grid.shape

# target size
crop_y, crop_x = 100, 100

# ---- Y: symmetric (keep center vertically) ----
y_start = max((ny - crop_y) // 2, 0)
y_end   = min(y_start + crop_y, ny)

# ---- X: keep LEFT side, cut from RIGHT ----
x_start = 0
x_end   = min(crop_x, nx)

# apply crop
u_grid = u_grid[y_start:y_end, x_start:x_end]
v_grid = v_grid[y_start:y_end, x_start:x_end]

ny, nx = u_grid.shape

# regenerate grid
x_axis = np.arange(nx)
y_axis = np.arange(ny)
X, Y = np.meshgrid(x_axis, y_axis)

print(f"Cropped shape: ({ny}, {nx})")

X_MIN, X_MAX = x_axis.min(), x_axis.max()
Y_MIN, Y_MAX = y_axis.min(), y_axis.max()

# ------------------ GEOMETRY ------------------
cx, cy = 50, 50
r = 25

cylinder = (X - cx)**2 + (Y - cy)**2 <= r**2

gap_w, gap_h = 60, 60
gap_x_min = cx - gap_h // 2
gap_x_max = cx + gap_h // 2
gap_y_min = cy - gap_w // 2
gap_y_max = cy + gap_w // 2

gap_mask = (X >= gap_x_min) & (X <= gap_x_max) & \
           (Y >= gap_y_min) & (Y <= gap_y_max)

gap_mask = gap_mask & (~cylinder)

# ------------------ TENSORS ------------------

# --- GAP (interior) → NS only ---
X_gap_t = torch.tensor(X[gap_mask], dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
Y_gap_t = torch.tensor(Y[gap_mask], dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
U_gap_t = torch.tensor(u_grid[gap_mask], dtype=torch.float32, device=device).unsqueeze(1)
V_gap_t = torch.tensor(v_grid[gap_mask], dtype=torch.float32, device=device).unsqueeze(1)

# --- GAP BOUNDARY (for BC) ---
boundary_mask = (
    ((X == gap_x_min) | (X == gap_x_max)) & (Y >= gap_y_min) & (Y <= gap_y_max)
) | (
    ((Y == gap_y_min) | (Y == gap_y_max)) & (X >= gap_x_min) & (X <= gap_x_max)
)

boundary_mask = boundary_mask & (~cylinder)

X_b_t = torch.tensor(X[boundary_mask], dtype=torch.float32, device=device).unsqueeze(1)
Y_b_t = torch.tensor(Y[boundary_mask], dtype=torch.float32, device=device).unsqueeze(1)

U_b_t = torch.tensor(u_grid[boundary_mask], dtype=torch.float32, device=device).unsqueeze(1)
V_b_t = torch.tensor(v_grid[boundary_mask], dtype=torch.float32, device=device).unsqueeze(1)

# --- DATA (outside gap, excluding boundary for clean separation) ---
data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask)

X_out_t = torch.tensor(X[data_mask], dtype=torch.float32, device=device).unsqueeze(1)
Y_out_t = torch.tensor(Y[data_mask], dtype=torch.float32, device=device).unsqueeze(1)

U_out_t = torch.tensor(u_grid[data_mask], dtype=torch.float32, device=device).unsqueeze(1)
V_out_t = torch.tensor(v_grid[data_mask], dtype=torch.float32, device=device).unsqueeze(1)

print(f"GAP interior (NS): {len(X_gap_t):,}")
print(f"GAP boundary (BC): {len(X_b_t):,}")
print(f"DATA (outside gap): {len(X_out_t):,}")


# ------------------ HYPERPARAMS ------------------
h = 64
nu = 1e-5

b_ns = 2000
b_data = 1000
b_bc = 500
b_gap = 500

w_ns = 1
w_bc = 50
w_data = 50
w_gap = 50


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

    ux, uy = grad(u, x), grad(u, y)
    vx, vy = grad(v, x), grad(v, y)

    uxx, uyy = grad(ux, x), grad(uy, y)
    vxx, vyy = grad(vx, x), grad(vy, y)

    px, py = grad(p, x), grad(p, y)

    cont = ux + vy
    momx = u*ux + v*uy - nu*(uxx + uyy) + px
    momy = u*vx + v*vy - nu*(vxx + vyy) + py

    return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)


# --- NS ONLY IN GAP ---
def NS_loss():
    idx = torch.randperm(len(X_gap_t), device=device)[:b_ns]
    return NS_loss_fn(X_gap_t[idx], Y_gap_t[idx])


# --- DATA LOSS (gap interior, as you intended) ---
def data_loss():
    if w_data == 0:
        return torch.tensor(0.0, device=device)

    idx = torch.randperm(len(X_gap_t), device=device)[:b_data]
    u_p, v_p, _ = model(X_gap_t[idx], Y_gap_t[idx]).split(1, dim=1)

    return torch.mean((u_p - U_gap_t[idx])**2) + \
           torch.mean((v_p - V_gap_t[idx])**2)


# --- CYLINDER BC ---
def bc_cylinder():
    a = 2 * np.pi * torch.rand(b_bc, 1, device=device)
    x = cx + r * torch.cos(a)
    y = cy + r * torch.sin(a)

    u, v, _ = model(x, y).split(1, dim=1)
    return torch.mean(u**2 + v**2)


# --- GAP BOUNDARY BC (NEW, separate) ---
def bc_gap():
    idx = torch.randperm(len(X_b_t), device=device)[:b_gap]

    u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)

    return torch.mean((u_p - U_b_t[idx])**2) + \
           torch.mean((v_p - V_b_t[idx])**2)

def loss():
    ns  = NS_loss()
    dat = data_loss()
    bc  = bc_cylinder()
    gap = bc_gap()

    total = w_ns*ns + w_data*dat + w_bc*bc + w_gap*gap

    return total, ns.item(), dat.item(), bc.item(), gap.item()

# ------------------ TRAINING + HISTORY ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

checkpoints = [0, 200, 500, 1000]

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

# Precompute full grid tensors
X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

for epoch in range(1001):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val, bcgap_val = loss()
    L.backward()
    optimizer.step()

    # ----- Metrics -----
    if epoch % 50 == 0:
        with torch.no_grad():
            u_flat, v_flat, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()

            # ground truth magnitude
            U_true = np.sqrt(u_grid**2 + v_grid**2)

            # full-domain L2
            rel_l2_full = np.linalg.norm(U_pred - U_true) / np.linalg.norm(U_true)

            # gap-only L2
            if np.linalg.norm(U_true[gap_mask]) > 0:
                rel_l2_gap = (
                    np.linalg.norm(U_pred[gap_mask] - U_true[gap_mask]) /
                    np.linalg.norm(U_true[gap_mask])
                )
            else:
                rel_l2_gap = 0.0

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

            U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()

        # shared normalization
        vmin = min(u_grid.min(), U_pred.min())
        vmax = max(u_grid.max(), U_pred.max())
        levels = np.linspace(vmin, vmax, 20)

        # ------------------ GROUND TRUTH ------------------
        cf_true = axs[0].contourf(
            X, Y, np.sqrt(u_grid**2 + v_grid**2),
            levels=levels,
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )

        axs[0].plot([gap_x_min, gap_x_max, gap_x_max, gap_x_min, gap_x_min],
                    [gap_y_min, gap_y_min, gap_y_max, gap_y_max, gap_y_min],
                    'k--', lw=1.5)

        circle0 = plt.Circle((cx, cy), r, color='k', fill=False, linestyle='--')
        axs[0].add_patch(circle0)

        axs[0].set_title(f"Ground Truth |U| (Epoch {epoch})")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].axis("equal")

        plt.colorbar(cf_true, ax=axs[0])

        # ------------------ PREDICTION ------------------
        cf_pred = axs[1].contourf(
            X, Y, U_pred,
            levels=levels,
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )

        axs[1].plot([gap_x_min, gap_x_max, gap_x_max, gap_x_min, gap_x_min],
                    [gap_y_min, gap_y_min, gap_y_max, gap_y_max, gap_y_min],
                    'k--', lw=1.5)

        circle1 = plt.Circle((cx, cy), r, color='k', fill=False, linestyle='--')
        axs[1].add_patch(circle1)

        axs[1].set_title(f"PINN Prediction |U| (Epoch {epoch})")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].axis("equal")

        plt.colorbar(cf_pred, ax=axs[1])

        plt.suptitle(f"Epoch {epoch} Comparison", fontsize=14)
        plt.tight_layout()
        plt.show()

# ------------------ FINAL PLOTS ------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(history["epoch"], history["loss_ns"],   label="NS")
axs[0].plot(history["epoch"], history["loss_data"], label="Data")
axs[0].plot(history["epoch"], history["loss_bc"],   label="BC (Cylinder)")
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