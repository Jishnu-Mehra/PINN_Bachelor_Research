"""
W&B hyperparameter sweep for the gridded (flow_data.npz) cylinder-gap PINN.

Usage:
    1. Set your API key as an environment variable (do NOT hardcode it):
         export WANDB_API_KEY=your_key_here
       or run `wandb login` once interactively.

    2. Launch the sweep controller:
         python train_pinn_sweep.py --init

       This prints a SWEEP_ID and starts one agent run in this process.

    3. To run more agents (parallel or sequential, same or other machines):
         python train_pinn_sweep.py --agent <SWEEP_ID>

    4. To just do a single normal (non-sweep) run with default config:
         python train_pinn_sweep.py --single
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "PINN_Bachelor_Research/Data/RealCylFlow/flow_data.npz"

# ------------------------------------------------------------------
# Fixed (non-swept) settings — same for every run
# ------------------------------------------------------------------
SEED = 42
RATIO = 0.08       # gap_thickness = ratio * d
EPOCHS = 3000       # bumped up from 2001 so slower-converging configs
                     # in the sweep aren't unfairly cut short

# ------------------------------------------------------------------
# Sweep search space — only the original tunable hyperparameters
# ------------------------------------------------------------------
sweep_config = {
    "method": "bayes",  # "grid", "random", or "bayes"
    "metric": {"name": "rel_l2_gap_final", "goal": "minimize"},
    "parameters": {
        "lr":              {"values": [1e-4, 5e-4, 1e-3, 2e-3]},
        "h":               {"values": [32, 64, 128]},
        "n_hidden_layers": {"values": [3, 5]},
        "w_ns":            {"min": 0.01, "max": 1.0},
        "w_bc":            {"min": 50.0, "max": 1000.0},
        "w_data":          {"min": 50.0, "max": 500.0},
        "w_gap":           {"min": 5.0, "max": 200.0},
        "b_ns":            {"values": [1000, 2000, 4000]},
        "b_data":          {"values": [5000, 10000, 20000]},
        "b_bc":            {"values": [2000, 5000]},
        "b_gap":           {"values": [500, 1000, 2000]},
    },
}


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


class PINN(nn.Module):
    def __init__(self, h, n_hidden_layers, x_min, x_max, y_min, y_max):
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        layers = [nn.Linear(2, h), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 3)]

        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        xn = normalise(x, self.x_min, self.x_max)
        yn = normalise(y, self.y_min, self.y_max)
        return self.net(torch.cat([xn, yn], dim=1))


def grad(f, x):
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True
    )[0]


# ------------------------------------------------------------------
# Data loading + masking (identical logic to your gridded script)
# ------------------------------------------------------------------
def load_data(ratio):
    data = np.load(DATA_PATH)
    u_grid = data["u"]
    v_grid = data["v"]
    X_full = data["x"]
    Y_full = data["y"]

    valid_mask = ~np.isnan(u_grid) & ~np.isnan(v_grid)

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

    u_grid = u_grid[y_start:y_end, x_start:x_end]
    v_grid = v_grid[y_start:y_end, x_start:x_end]
    X = X_full[y_start:y_end, x_start:x_end]
    Y = Y_full[y_start:y_end, x_start:x_end]
    valid_mask = valid_mask[y_start:y_end, x_start:x_end]

    cx, cy = 0, 0
    d = 125
    r = d / 2
    gap_thickness = ratio * d

    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    cylinder = dist2 <= r**2

    r_inner = r
    r_outer = r + gap_thickness

    gap_mask = (dist2 >= r_inner**2) & (dist2 <= r_outer**2)
    gap_mask = gap_mask & (~cylinder)

    X_MIN, X_MAX = X.min(), X.max()
    tol = (X_MAX - X_MIN) / 500
    dist = np.sqrt(dist2)

    boundary_mask = np.abs(dist - r_outer) < tol
    boundary_mask = boundary_mask & (~cylinder) & (~gap_mask)

    data_mask = (~cylinder) & (~gap_mask) & (~boundary_mask)

    gap_mask &= valid_mask
    boundary_mask &= valid_mask
    data_mask &= valid_mask

    return dict(
        X=X, Y=Y, u_grid=u_grid, v_grid=v_grid,
        cylinder=cylinder, gap_mask=gap_mask,
        boundary_mask=boundary_mask, data_mask=data_mask,
        cx=cx, cy=cy, r=r, gap_thickness=gap_thickness,
        valid_mask=valid_mask,
    )


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)


# ------------------------------------------------------------------
# Training loop for a single sweep run
# ------------------------------------------------------------------
def train():
    run = wandb.init()
    cfg = wandb.config

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = load_data(RATIO)
    X, Y = d["X"], d["Y"]
    u_grid, v_grid = d["u_grid"], d["v_grid"]
    cylinder, gap_mask = d["cylinder"], d["gap_mask"]
    boundary_mask, data_mask = d["boundary_mask"], d["data_mask"]
    cx, cy, r, gap_thickness = d["cx"], d["cy"], d["r"], d["gap_thickness"]

    X_MIN, X_MAX = X.min(), X.max()
    Y_MIN, Y_MAX = Y.min(), Y.max()

    X_gap_t = to_tensor(X[gap_mask]).requires_grad_(True)
    Y_gap_t = to_tensor(Y[gap_mask]).requires_grad_(True)

    X_b_t = to_tensor(X[boundary_mask])
    Y_b_t = to_tensor(Y[boundary_mask])
    U_b_t = to_tensor(u_grid[boundary_mask])
    V_b_t = to_tensor(v_grid[boundary_mask])

    X_out_t = to_tensor(X[data_mask])
    Y_out_t = to_tensor(Y[data_mask])
    U_out_t = to_tensor(u_grid[data_mask])
    V_out_t = to_tensor(v_grid[data_mask])

    N_wall = 30000
    theta_wall = 2 * np.pi * np.random.rand(N_wall)
    rr = r + gap_thickness * (np.random.rand(N_wall) ** 2)
    x_wall_ns = cx + rr * np.cos(theta_wall)
    y_wall_ns = cy + rr * np.sin(theta_wall)
    X_wall_ns_t = to_tensor(x_wall_ns).requires_grad_(True)
    Y_wall_ns_t = to_tensor(y_wall_ns).requires_grad_(True)

    nu = 1.5e-5

    model = PINN(cfg.h, cfg.n_hidden_layers, X_MIN, X_MAX, Y_MIN, Y_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    def NS_loss_fn(x, y):
        u, v, p = model(x, y).split(1, dim=1)
        ux = grad(u, x); uy = grad(u, y)
        vx = grad(v, x); vy = grad(v, y)
        uxx = grad(ux, x); uyy = grad(uy, y)
        vxx = grad(vx, x); vyy = grad(vy, y)
        px = grad(p, x); py = grad(p, y)
        cont = ux + vy
        momx = u * ux + v * uy - nu * (uxx + uyy) + px
        momy = u * vx + v * vy - nu * (vxx + vyy) + py
        return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)

    def NS_loss():
        idx1 = torch.randperm(len(X_gap_t), device=device)[: cfg.b_ns // 2]
        idx2 = torch.randperm(len(X_wall_ns_t), device=device)[: cfg.b_ns // 2]
        loss_gap = NS_loss_fn(X_gap_t[idx1], Y_gap_t[idx1])
        loss_wall = NS_loss_fn(X_wall_ns_t[idx2], Y_wall_ns_t[idx2])
        return loss_gap + 5 * loss_wall

    def data_loss():
        idx = torch.randperm(len(X_out_t), device=device)[: cfg.b_data]
        u_p, v_p, _ = model(X_out_t[idx], Y_out_t[idx]).split(1, dim=1)
        return torch.mean((u_p - U_out_t[idx]) ** 2) + torch.mean((v_p - V_out_t[idx]) ** 2)

    def bc_cylinder():
        a = 2 * np.pi * torch.rand(cfg.b_bc, 1, device=device)
        xc = cx + r * torch.cos(a)
        yc = cy + r * torch.sin(a)
        u, v, _ = model(xc, yc).split(1, dim=1)
        return torch.mean(u**2 + v**2)

    def bc_gap():
        idx = torch.randperm(len(X_b_t), device=device)[: cfg.b_gap]
        u_p, v_p, _ = model(X_b_t[idx], Y_b_t[idx]).split(1, dim=1)
        return torch.mean((u_p - U_b_t[idx]) ** 2) + torch.mean((v_p - V_b_t[idx]) ** 2)

    def total_loss():
        ns = NS_loss()
        dat = data_loss()
        bc = bc_cylinder()
        gap = bc_gap()
        total = cfg.w_ns * ns + cfg.w_data * dat + cfg.w_bc * bc + cfg.w_gap * gap
        return total, ns.item(), dat.item(), bc.item(), gap.item()

    X_flat_t = to_tensor(X.flatten())
    Y_flat_t = to_tensor(Y.flatten())
    U_true = np.sqrt(u_grid**2 + v_grid**2)

    rel_l2_gap_final = float("inf")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        L, ns_val, dat_val, bc_val, bcgap_val = total_loss()
        L.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                u_p, v_p, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
                U_pred = torch.sqrt(u_p**2 + v_p**2).reshape(X.shape).cpu().numpy()

            valid = (~cylinder) & d["valid_mask"] & (~np.isnan(U_pred))
            rel_l2_full = np.linalg.norm(U_pred[valid] - U_true[valid]) / np.linalg.norm(U_true[valid])

            gap_valid = gap_mask & valid
            if gap_valid.sum() > 0:
                rel_l2_gap = np.linalg.norm(U_pred[gap_valid] - U_true[gap_valid]) / np.linalg.norm(U_true[gap_valid])
            else:
                rel_l2_gap = float("nan")

            rel_l2_gap_final = rel_l2_gap

            wandb.log({
                "epoch": epoch,
                "loss_total": L.item(),
                "loss_ns": ns_val,
                "loss_data": dat_val,
                "loss_bc": bc_val,
                "loss_bc_gap": bcgap_val,
                "rel_l2_full": rel_l2_full,
                "rel_l2_gap": rel_l2_gap,
                "lr": optimizer.param_groups[0]["lr"],
            })

    wandb.summary["rel_l2_gap_final"] = rel_l2_gap_final
    wandb.summary["rel_l2_full_final"] = rel_l2_full
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="create sweep + run one agent here")
    parser.add_argument("--agent", type=str, default=None, help="join an existing sweep by ID")
    parser.add_argument("--single", action="store_true", help="single run, no sweep")
    parser.add_argument("--count", type=int, default=20, help="number of runs this agent executes")
    args = parser.parse_args()

    ENTITY = "jishnu-mehra-delft-university-of-technology"
    PROJECT = "pinn-cylinder-gap"

    if args.single:
        wandb.init(entity=ENTITY, project=PROJECT, config={
            "lr": 1e-3, "h": 64, "n_hidden_layers": 3,
            "w_ns": 0.1, "w_bc": 500, "w_data": 100, "w_gap": 50,
            "b_ns": 2000, "b_data": 10000, "b_bc": 5000, "b_gap": 1000,
        })
        # wandb.init() above already set up the run; train() reuses it
        # (its own wandb.init() call becomes a no-op inside an active run).
        train()

    elif args.init:
        sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
        print(f"Sweep created: {sweep_id}")
        print(f"Run more agents with: python {os.path.basename(__file__)} --agent {sweep_id}")
        wandb.agent(sweep_id, function=train, count=args.count)

    elif args.agent:
        wandb.agent(args.agent, function=train, entity=ENTITY, project=PROJECT, count=args.count)

    else:
        parser.print_help()