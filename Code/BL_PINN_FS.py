import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

x_axis = np.load("PINN_Bachelor_Research/Data/x.npy")
y_axis = np.load("PINN_Bachelor_Research/Data/y.npy")
u_data = np.load("PINN_Bachelor_Research/Data/u.npy")
v_data = np.load("PINN_Bachelor_Research/Data/v.npy")

V_CLIP = np.percentile(np.abs(v_data), 99.9)
v_data = np.clip(v_data, -V_CLIP, V_CLIP)

X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")

X_MIN, X_MAX = x_axis.min(), x_axis.max()
Y_MIN, Y_MAX = y_axis.min(), y_axis.max()
gap_x_min, gap_x_max = X_MIN + 1, X_MAX - 1
gap_y_min = Y_MIN
gap_y_max = 0.1 * Y_MAX

gap_mask = (Y >= gap_y_min) & (Y <= gap_y_max) & (X >= gap_x_min) & (X <= gap_x_max)

above_mask = Y > gap_y_max
X_out_t = torch.tensor(X[above_mask],      dtype=torch.float32, device=device).unsqueeze(1)
Y_out_t = torch.tensor(Y[above_mask],      dtype=torch.float32, device=device).unsqueeze(1)
U_out_t = torch.tensor(u_data[above_mask], dtype=torch.float32, device=device).unsqueeze(1)
V_out_t = torch.tensor(v_data[above_mask], dtype=torch.float32, device=device).unsqueeze(1)

print(f"Data pool (y > {gap_y_max:.4f}): {len(X_out_t):,} points")

X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

torch.manual_seed(0)
N_gap_eval = 2000
X_gap_eval = (gap_x_min + (gap_x_max - gap_x_min) * torch.rand(N_gap_eval, 1)).to(device)
Y_gap_eval = (gap_y_min + (gap_y_max - gap_y_min) * torch.rand(N_gap_eval, 1)**2).to(device)

def sample_background(x_t, y_t):
    x_np = x_t.detach().cpu().numpy().ravel()
    y_np = y_t.detach().cpu().numpy().ravel()

    def nearest_idx(axis, vals):
        idx = np.searchsorted(axis, vals)
        idx = np.clip(idx, 0, len(axis) - 1)
        ip  = np.clip(idx - 1, 0, len(axis) - 1)
        return np.where(np.abs(axis[ip] - vals) <= np.abs(axis[idx] - vals), ip, idx)

    ix = nearest_idx(x_axis, x_np)
    iy = nearest_idx(y_axis, y_np)
    u  = torch.tensor(u_data[iy, ix], dtype=torch.float32, device=device).unsqueeze(1)
    v  = torch.tensor(v_data[iy, ix], dtype=torch.float32, device=device).unsqueeze(1)
    return u, v


def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


class BL_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x, y):
        xn = normalise(x, X_MIN, X_MAX)
        yn = normalise(y, Y_MIN, Y_MAX)
        return self.net(torch.cat([xn, yn], dim=1))

def grad(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                               create_graph=True, retain_graph=True)[0]

model = BL_PINN().to(device)
nu = 1e-5

def data_loss():
    idx = torch.randperm(len(X_out_t), device=device)[:500]
    u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)
    return torch.mean((u_p - U_out_t[idx])**2) + torch.mean((v_p - V_out_t[idx])**2)

def NS_loss_fn(x, y):
    u, v, p = model(x, y).split(1, dim=1)
    ux, uy = grad(u, x), grad(u, y)
    vy     = grad(v, y)
    uyy    = grad(uy, y)
    px, py = grad(p, x), grad(p, y)
    cont = ux + vy
    momx = u*ux + v*uy - nu*uyy + (1/1.225)*px
    momy = (1/1.225)*py
    return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)

def NS_loss():
    N = 2000
    x = (gap_x_min + (gap_x_max - gap_x_min) * torch.rand(N, 1, device=device)).requires_grad_(True)
    y = (gap_y_min + (gap_y_max - gap_y_min) * torch.rand(N, 1, device=device)**2).requires_grad_(True)
    return NS_loss_fn(x, y)

def gap_bc(y_val):
    x = gap_x_min + (gap_x_max - gap_x_min) * torch.rand(50, 1, device=device)
    y = torch.ones_like(x) * y_val
    u_p, v_p, _ = model(x, y).split(1, dim=1)
    u_inf, v_inf = sample_background(x, y)
    return torch.mean((u_p - u_inf)**2) + torch.mean((v_p - v_inf)**2)

def loss():
    ns   = NS_loss()
    dat  = data_loss()
    bc_b = gap_bc(gap_y_min)
    bc_t = gap_bc(gap_y_max)
    total = 0.1*ns + 100*dat + 100*bc_b + 100*bc_t
    return total, ns.item(), dat.item(), (bc_b + bc_t).item()

# --- Training ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoints = [1, 200, 500, 1000]

history = {
    "epoch":       [],
    "loss_total":  [],
    "loss_ns":     [],
    "loss_data":   [],
    "loss_bc":     [], 
    "loss_ns_gap": [], 
    "rel_l2_full": [],
    "rel_l2_gap":  [],
}

for epoch in range(1001):
    optimizer.zero_grad()
    L, ns_val, dat_val, bc_val = loss()
    L.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            u_flat, _, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = u_flat.reshape(X.shape).cpu().numpy()

            rel_l2_full = (np.linalg.norm(U_pred - u_data) /
                           np.linalg.norm(u_data))
            rel_l2_gap  = (np.linalg.norm(U_pred[gap_mask] - u_data[gap_mask]) /
                           np.linalg.norm(u_data[gap_mask]))

        x_g = X_gap_eval.clone().requires_grad_(True)
        y_g = Y_gap_eval.clone().requires_grad_(True)
        ns_gap = NS_loss_fn(x_g, y_g).item()

        history["epoch"].append(epoch)
        history["loss_total"].append(L.item())
        history["loss_ns"].append(ns_val)
        history["loss_data"].append(dat_val)
        history["loss_bc"].append(bc_val)
        history["loss_ns_gap"].append(ns_gap)
        history["rel_l2_full"].append(rel_l2_full)
        history["rel_l2_gap"].append(rel_l2_gap)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {history['loss_total'][-1]:.3e} | "
              f"NS {history['loss_ns'][-1]:.3e} | "
              f"Data {history['loss_data'][-1]:.3e} | "
              f"BC {history['loss_bc'][-1]:.3e} | "
              f"L2 gap {history['rel_l2_gap'][-1]:.4e}")

    if epoch in checkpoints:
        with torch.no_grad():
            u_flat, _, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
            U_pred = u_flat.reshape(X.shape).cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(14, 3))
        cf_true = axs[0].contourf(X, Y, u_data, levels=20, cmap="jet")
        axs[0].plot([gap_x_min, gap_x_max, gap_x_max, gap_x_min, gap_x_min],
                    [gap_y_min, gap_y_min, gap_y_max, gap_y_max, gap_y_min], 'k--', lw=1.5)
        axs[0].set_title(f"Ground Truth (Epoch {epoch})")
        axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
        plt.colorbar(cf_true, ax=axs[0])

        cf_pred = axs[1].contourf(X, Y, U_pred, levels=20, cmap="jet")
        axs[1].plot([gap_x_min, gap_x_max, gap_x_max, gap_x_min, gap_x_min],
                    [gap_y_min, gap_y_min, gap_y_max, gap_y_max, gap_y_min], 'k--', lw=1.5)
        axs[1].set_title(f"PINN Prediction (Epoch {epoch})")
        axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
        plt.colorbar(cf_pred, ax=axs[1])
        plt.suptitle(f"Epoch {epoch} Comparison", fontsize=14)
        plt.tight_layout()
        plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(history["epoch"], history["loss_ns"],   color="orange", label="NS (gap)")
axs[0].plot(history["epoch"], history["loss_data"],  color="blue",   label="Data")
axs[0].plot(history["epoch"], history["loss_bc"],    color="green",  label="BC (top+bottom)")
axs[0].set_yscale("log")
axs[0].set_xlabel("Epoch"); axs[0].set_title("Loss components (unweighted)")
axs[0].legend(); axs[0].grid(True, which="both", ls="--", alpha=0.5)

axs[1].plot(history["epoch"], history["rel_l2_gap"],  color="red",  label="Gap only")
axs[1].plot(history["epoch"], history["rel_l2_full"], color="blue", label="Full domain", linestyle="--")
axs[1].set_yscale("log")
axs[1].set_xlabel("Epoch"); axs[1].set_title("Relative $L_2$ Error")
axs[1].legend(); axs[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()