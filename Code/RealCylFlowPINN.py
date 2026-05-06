# import numpy as np
# import re

# file_path = "PINN_Bachelor_Research/Data/RealCylFlow/Cylinder_D_125mm_Uinf_5ms.tp"

# with open(file_path, "r") as f:
#     lines = f.readlines()

# # --- Extract grid size ---
# for line in lines:
#     if "ZONE" in line:
#         I = int(re.search(r"I=(\d+)", line).group(1))
#         J = int(re.search(r"J=(\d+)", line).group(1))
#         break

# # --- Keep ONLY numeric lines ---
# data_lines = []
# for line in lines:
#     # skip header / metadata
#     if any(key in line for key in ["TITLE", "VARIABLES", "ZONE", "STRANDID"]):
#         continue
#     if line.strip() == "":
#         continue

#     # keep only lines that start with a number (or minus sign)
#     if line.strip()[0] in "-0123456789":
#         data_lines.append(line)

# # --- Load safely ---
# data = np.loadtxt(data_lines)

# print("Loaded shape:", data.shape)  # should be (I*J, num_variables)

# x = data[:, 0]
# y = data[:, 1]
# u = data[:, 2]
# v = data[:, 3]
# Vmag = data[:, 5]

# nx, ny = I, J

# x_grid = x.reshape(ny, nx)
# y_grid = y.reshape(ny, nx)
# u_grid = u.reshape(ny, nx)
# v_grid = v.reshape(ny, nx)
# V_grid = Vmag.reshape(ny, nx)

# np.savez("flow_data.npz",
#          x=x_grid,
#          y=y_grid,
#          u=u_grid,
#          v=v_grid,
#          V=V_grid)

# import numpy as np
# import matplotlib.pyplot as plt

# # load
# data = np.load("PINN_Bachelor_Research/Data/RealCylFlow/flow_data.npz")

# u = data["u"]
# v = data["v"]
# U = data["V"]
# x = data["x"]
# y = data["y"]

# plt.figure(figsize=(6, 4))
# cf = plt.contourf(x, y, U, levels=20, cmap="jet")
# plt.gca().set_aspect('equal')

# plt.title("Velocity Magnitude |U| (with coordinates)")
# plt.xlabel("x")
# plt.ylabel("y")

# plt.colorbar(cf)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------ LOAD ------------------
data = np.load("PINN_Bachelor_Research/Data/RealCylFlow/flow_data.npz")

u_grid = data["u"]
v_grid = data["v"]
X_full = data["x"]
Y_full = data["y"]

# ------------------ GLOBAL VALID MASK ------------------
valid_mask = ~np.isnan(u_grid) & ~np.isnan(v_grid)

# ------------------ PHYSICAL CROPPING ------------------
x_min = np.min(X_full)
x_max_crop = 300

y_min_crop = -100
y_max_crop = 100

x_mask = (X_full[0, :] >= x_min) & (X_full[0, :] <= x_max_crop)
y_mask = (Y_full[:, 0] >= y_min_crop) & (Y_full[:, 0] <= y_max_crop)

x_idx = np.where(x_mask)[0]
y_idx = np.where(y_mask)[0]

x_start, x_end = x_idx[0], x_idx[-1] + 1
y_start, y_end = y_idx[0], y_idx[-1] + 1

# apply crop
u_grid = u_grid[y_start:y_end, x_start:x_end]
v_grid = v_grid[y_start:y_end, x_start:x_end]
X = X_full[y_start:y_end, x_start:x_end]
Y = Y_full[y_start:y_end, x_start:x_end]
valid_mask = valid_mask[y_start:y_end, x_start:x_end]

ny, nx = u_grid.shape
print(f"Cropped shape: ({ny}, {nx})")

# bounds
X_MIN, X_MAX = X.min(), X.max()
Y_MIN, Y_MAX = Y.min(), Y.max()

# ------------------ GEOMETRY ------------------
cx, cy = 0, 0
r = 125 / 2
gap_thickness = 10

dist2 = (X - cx)**2 + (Y - cy)**2
cylinder = dist2 <= r**2

r_inner = r
r_outer = r + gap_thickness

gap_mask = (dist2 >= r_inner**2) & (dist2 <= r_outer**2)
gap_mask = gap_mask & (~cylinder)

# boundary
tol = (X_MAX - X_MIN) / 500
dist = np.sqrt(dist2)

boundary_mask = (np.abs(dist - r_outer) < tol)
boundary_mask = boundary_mask & (~cylinder) & (~gap_mask)

# data outside
data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask)

# ------------------ APPLY VALID MASK ------------------
gap_mask      &= valid_mask
boundary_mask &= valid_mask
data_mask     &= valid_mask

# ------------------ TENSORS ------------------
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)

X_gap_t = to_tensor(X[gap_mask]).requires_grad_(True)
Y_gap_t = to_tensor(Y[gap_mask]).requires_grad_(True)

U_gap_t = to_tensor(u_grid[gap_mask])
V_gap_t = to_tensor(v_grid[gap_mask])

X_b_t = to_tensor(X[boundary_mask])
Y_b_t = to_tensor(Y[boundary_mask])

U_b_t = to_tensor(u_grid[boundary_mask])
V_b_t = to_tensor(v_grid[boundary_mask])

X_out_t = to_tensor(X[data_mask])
Y_out_t = to_tensor(Y[data_mask])

U_out_t = to_tensor(u_grid[data_mask])
V_out_t = to_tensor(v_grid[data_mask])

print(f"GAP interior: {len(X_gap_t)}")
print(f"GAP boundary: {len(X_b_t)}")
print(f"DATA: {len(X_out_t)}")


# ------------------ HYPERPARAMS ------------------
h = 64
nu = 1e-5

b_ns = 2000
b_data = 10000
b_bc = 500
b_gap = 500

w_ns = 0.1
w_bc = 50
w_data = 100
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

    idx = torch.randperm(len(X_out_t), device=device)[:b_data]
    u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)

    return torch.mean((u_p - U_out_t[idx])**2) + \
           torch.mean((v_p - V_out_t[idx])**2)


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

checkpoints = [0, 200, 500, 1000, 2000]

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

for epoch in range(2001):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val, bcgap_val = loss()
    L.backward()
    optimizer.step()

    # ----- Metrics -----
    if epoch % 50 == 0:
        with torch.no_grad():
            u, v, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = torch.sqrt(u**2 + v**2).reshape(X.shape).cpu().numpy()

        U_true = np.sqrt(u_grid**2 + v_grid**2)

        valid = (~cylinder) & valid_mask & (~np.isnan(U_pred))

        rel_l2_full = np.linalg.norm(U_pred[valid] - U_true[valid]) / np.linalg.norm(U_true[valid])

        gap_valid = gap_mask & valid
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

            U_pred = torch.sqrt(u_flat**2 + v_flat**2).reshape(X.shape).cpu().numpy()

        U_true_mag = np.sqrt(u_grid**2 + v_grid**2)

        # mask out cylinder + NaNs
        valid_mask = (~cylinder) & (~np.isnan(U_true_mag))

        vmin = np.min(U_true_mag[valid_mask])
        vmax = np.max(U_true_mag[valid_mask])

        levels = np.linspace(vmin, vmax, 20)

        # ------------------ GROUND TRUTH ------------------
        cf_true = axs[0].contourf(
            X, Y, np.sqrt(u_grid**2 + v_grid**2),
            levels=levels,
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )
        circle0 = plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--')
        axs[0].add_patch(circle0)

        circle1 = plt.Circle((cx, cy), r, color='k', fill=False, linestyle='--')
        axs[0].add_patch(circle1)

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

        circle2 = plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--')
        axs[1].add_patch(circle2)

        circle3 = plt.Circle((cx, cy), r, color='k', fill=False, linestyle='--')
        axs[1].add_patch(circle3)

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

model.eval()

with torch.no_grad():
    u, v, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
    U_pred = torch.sqrt(u**2 + v**2).reshape(X.shape).cpu().numpy()

U_true = np.sqrt(u_grid**2 + v_grid**2)

U_hybrid = U_true.copy()
U_hybrid[gap_mask] = U_pred[gap_mask]

valid = (~cylinder) & (~np.isnan(U_true)) & (~np.isnan(U_pred))

vmin = np.min(U_true[valid])
vmax = np.max(U_true[valid])

U_plot = np.ma.array(U_hybrid, mask=cylinder)
plt.contourf(X, Y, U_plot, levels=20, cmap="jet", vmin=vmin, vmax=vmax)
plt.gca().set_aspect('equal')
circle = plt.Circle((cx, cy), r, color='k', fill=False, linestyle='--')
plt.gca().add_patch(circle)

circlee = plt.Circle((cx, cy), r+gap_thickness, color='k', fill=False, linestyle='--')
plt.gca().add_patch(circlee)

plt.title("PINN in Gap + Ground Truth Outside")
plt.colorbar()
plt.show()