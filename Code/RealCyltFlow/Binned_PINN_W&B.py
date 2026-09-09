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

CHANGELOG (this version):
  - Sweep objective switched from an unweighted, unit-mismatched sum of
    raw loss terms ("physical_loss_sum_final") to "gap_mass_div_final":
    the RMS continuity (mass-conservation) residual evaluated ONLY on
    points inside the gap. This does not require gap ground truth (no
    leakage) and is dimensionally consistent (same physical quantity
    everywhere), unlike summing NS-residual + velocity-MSE terms.
  - physical_loss_sum_final is kept, but now computed as a NORMALIZED
    sum (each term divided by a fixed reference scale) instead of a raw
    sum, so one term can't dominate purely due to units/scale. Logged
    as a diagnostic, not the sweep target.
  - Added a skin-friction sanity check against Thom's laminar
    boundary-layer estimate C_Df = 4.0/sqrt(Re) at Re=41667. This is an
    order-of-magnitude reference only (see notes at definition) — not a
    tight target, since it excludes pressure/form drag and is only
    strictly valid ahead of separation.
  - All headline scalars are now also wandb.log()-ed (not just dumped
    into wandb.summary), and grouped with define_metric so they're easy
    to find as normal chart panels instead of only appearing in the
    runs table.
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

# Flow conditions (from postprocess script comments):
#   Re = 41667, 8 mm bin w/ 75% overlap, 0.5 mm bubbles.
RE = 41667.0
RHO = 1.225
NU = 1.5e-5
U_INF = 5.0  # matches postprocess script's U_inf

# Thom's laminar boundary-layer estimate for cylinder skin-friction drag
# coefficient (front-of-cylinder, pre-separation, laminar BL only):
#     C_Df = 4.0 / sqrt(Re)
# This is NOT the total drag coefficient (C_D ~ 1.2 in the subcritical
# regime 2e4 < Re < 3e5 per ESDU-type data — dominated by pressure/form
# drag, not friction, at this Re) and it is NOT valid past the
# separation point. Use only as an order-of-magnitude sanity check on
# the peak/average magnitude of the predicted C_f(theta) curve, never
# as a tight fit target.
CF_THOM_REFERENCE = 4.0 / np.sqrt(RE)  # ~= 0.0196

# ------------------------------------------------------------------
# Reference scales for normalizing the physical-loss diagnostic sum.
# Pick these from a representative baseline run (e.g. the --single
# defaults) so no single raw term dominates purely due to units.
# Update these if your baseline run's loss magnitudes change a lot.
# ------------------------------------------------------------------
REF_SCALE_NS = 0.05
REF_SCALE_DATA = 0.02
REF_SCALE_BC = 0.001
REF_SCALE_BCGAP = 0.005

# ------------------------------------------------------------------
# Sweep search space — only the original tunable hyperparameters
# ------------------------------------------------------------------
sweep_config = {
    "method": "bayes",  # "grid", "random", or "bayes"
    # Optimize the gap-region mass-conservation residual: it is
    # dimensionally consistent (always a continuity-residual^2 in the
    # same units, unlike summing velocity-MSE + PDE-residual terms) and
    # does not require gap ground truth, so it's a genuine physical
    # self-consistency check specific to the region the PINN is meant
    # to infer without labels.
    "metric": {"name": "gap_mass_div_final", "goal": "minimize"},
    "parameters": {
        "lr":              {"values": [1e-4, 5e-4, 1e-3, 2e-3]},
        "h":               {"values": [32, 64, 128]},
        "n_hidden_layers": {"values": [3, 5]},
        "w_ns":            {"min": 0.01, "max": 1.0},
        "w_bc":            {"min": 50.0, "max": 1000.0},
        "w_data":          {"min": 10.0, "max": 300.0},
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

    # Group metrics so the important ones are easy to find as chart
    # panels (not just runs-table columns), and separate the noisy
    # per-epoch training curves from the headline sweep metric.
    wandb.define_metric("epoch")
    wandb.define_metric("gap_mass_div_final", summary="min")
    wandb.define_metric("physical_loss_sum_norm_final", summary="min")
    wandb.define_metric("cf_thom_ratio_final", summary="min")
    wandb.define_metric("loss_*", step_metric="epoch")
    wandb.define_metric("rel_l2_*", step_metric="epoch")

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

    # --- Fixed evaluation sets for the sweep metric ---
    # Separate from the randomly-batched training tensors above so every
    # trial in the sweep is scored on identical points, independent of
    # batch size (b_ns/b_data/b_bc/b_gap) and independent of the loss
    # weights (w_ns/w_data/w_bc/w_gap) used during training. This makes
    # the sweep metric a genuine cross-trial comparison rather than an
    # artifact of "smaller weights -> smaller weighted total".
    eval_rng = np.random.RandomState(SEED)

    EVAL_N_GAP = min(4000, len(X[gap_mask]))
    eval_gap_idx = eval_rng.choice(len(X[gap_mask]), size=EVAL_N_GAP, replace=False)
    X_gap_eval = to_tensor(X[gap_mask][eval_gap_idx])
    Y_gap_eval = to_tensor(Y[gap_mask][eval_gap_idx])

    EVAL_N_WALL = 4000
    theta_eval = 2 * np.pi * eval_rng.rand(EVAL_N_WALL)
    rr_eval = r + gap_thickness * (eval_rng.rand(EVAL_N_WALL) ** 2)
    X_wall_eval = to_tensor(cx + rr_eval * np.cos(theta_eval))
    Y_wall_eval = to_tensor(cy + rr_eval * np.sin(theta_eval))

    EVAL_N_DATA = min(10000, len(X[data_mask]))
    eval_data_idx = eval_rng.choice(len(X[data_mask]), size=EVAL_N_DATA, replace=False)
    X_out_eval = to_tensor(X[data_mask][eval_data_idx])
    Y_out_eval = to_tensor(Y[data_mask][eval_data_idx])
    U_out_eval = to_tensor(u_grid[data_mask][eval_data_idx])
    V_out_eval = to_tensor(v_grid[data_mask][eval_data_idx])

    EVAL_N_BC = 4000
    theta_bc_eval = 2 * np.pi * eval_rng.rand(EVAL_N_BC)
    X_cyl_eval = to_tensor(cx + r * np.cos(theta_bc_eval))
    Y_cyl_eval = to_tensor(cy + r * np.sin(theta_bc_eval))

    EVAL_N_GAPBC = min(4000, len(X[boundary_mask]))
    eval_bc_idx = eval_rng.choice(len(X[boundary_mask]), size=EVAL_N_GAPBC, replace=False)
    X_b_eval = to_tensor(X[boundary_mask][eval_bc_idx])
    Y_b_eval = to_tensor(Y[boundary_mask][eval_bc_idx])
    U_b_eval = to_tensor(u_grid[boundary_mask][eval_bc_idx])
    V_b_eval = to_tensor(v_grid[boundary_mask][eval_bc_idx])

    # --- Fixed points on the cylinder wall itself, for the C_f(theta)
    # sanity check against the Thom reference. Independent of the gap
    # (the no-slip wall BC is enforced regardless of gap uncertainty).
    EVAL_N_CF = 360
    theta_cf_eval = np.linspace(0, 2 * np.pi, EVAL_N_CF)
    X_cf_eval = to_tensor(cx + r * np.cos(theta_cf_eval)).requires_grad_(True)
    Y_cf_eval = to_tensor(cy + r * np.sin(theta_cf_eval)).requires_grad_(True)
    nx_cf = torch.tensor(np.cos(theta_cf_eval), dtype=torch.float32, device=device).unsqueeze(1)
    ny_cf = torch.tensor(np.sin(theta_cf_eval), dtype=torch.float32, device=device).unsqueeze(1)
    tx_cf = -ny_cf
    ty_cf = nx_cf

    nu = NU

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

    def evaluate_gap_mass_divergence():
        """RMS continuity residual (|div(u)|^2, averaged) evaluated ONLY
        on points inside the gap region. This requires no gap ground
        truth (the PINN never sees labels there) and is dimensionally
        consistent across all sweep trials (always a squared-divergence
        quantity in the same units), unlike summing NS-residual with
        velocity-MSE terms of different physical units. Low divergence
        in the gap means the PINN's inferred gap flow is internally
        physically self-consistent (mass-conserving), independent of
        whether it matches any withheld reference."""
        xg = X_gap_eval.clone().requires_grad_(True)
        yg = Y_gap_eval.clone().requires_grad_(True)
        u, v, _ = model(xg, yg).split(1, dim=1)
        ux = grad(u, xg)
        vy = grad(v, yg)
        div = ux + vy
        return torch.mean(div**2).item()

    def evaluate_cf_theta():
        """Skin-friction coefficient distribution on the cylinder wall,
        for a sanity check only (see CF_THOM_REFERENCE definition for
        why this is order-of-magnitude, not a tight fit target)."""
        u, v, _ = model(X_cf_eval, Y_cf_eval).split(1, dim=1)
        ux = grad(u, X_cf_eval)
        uy = grad(u, Y_cf_eval)
        vx = grad(v, X_cf_eval)
        vy = grad(v, Y_cf_eval)
        du_dn = ux * nx_cf + uy * ny_cf
        dv_dn = vx * nx_cf + vy * ny_cf
        dut_dn = du_dn * tx_cf + dv_dn * ty_cf
        tau_w = RHO * NU * dut_dn
        cf = 2 * tau_w / (RHO * U_INF**2)
        cf_mean_abs = torch.mean(torch.abs(cf)).item()
        return cf_mean_abs

    def evaluate_physical_loss():
        """Normalized sum of the four raw loss components on the fixed
        eval sets. Each term is divided by a fixed reference scale
        (REF_SCALE_*, taken from a representative baseline run) before
        summing, so no single term dominates purely because of its
        units/magnitude. This is a diagnostic only — the sweep objective
        is gap_mass_div_final, computed separately above."""
        xg = X_gap_eval.clone().requires_grad_(True)
        yg = Y_gap_eval.clone().requires_grad_(True)
        xw = X_wall_eval.clone().requires_grad_(True)
        yw = Y_wall_eval.clone().requires_grad_(True)

        ns_eval = (NS_loss_fn(xg, yg) + 5 * NS_loss_fn(xw, yw)).item()

        with torch.no_grad():
            u_p, v_p, _ = model(X_out_eval, Y_out_eval).split(1, dim=1)
            dat_eval = (torch.mean((u_p - U_out_eval) ** 2) + torch.mean((v_p - V_out_eval) ** 2)).item()

            u_c, v_c, _ = model(X_cyl_eval, Y_cyl_eval).split(1, dim=1)
            bc_eval = torch.mean(u_c**2 + v_c**2).item()

            u_b, v_b, _ = model(X_b_eval, Y_b_eval).split(1, dim=1)
            bcgap_eval = (torch.mean((u_b - U_b_eval) ** 2) + torch.mean((v_b - V_b_eval) ** 2)).item()

        norm_sum = (
            ns_eval / REF_SCALE_NS
            + dat_eval / REF_SCALE_DATA
            + bc_eval / REF_SCALE_BC
            + bcgap_eval / REF_SCALE_BCGAP
        )
        raw_sum = ns_eval + dat_eval + bc_eval + bcgap_eval
        return norm_sum, raw_sum, ns_eval, dat_eval, bc_eval, bcgap_eval

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

    model.eval()
    gap_mass_div = evaluate_gap_mass_divergence()
    cf_mean_abs = evaluate_cf_theta()
    norm_sum, raw_sum, ns_e, dat_e, bc_e, bcgap_e = evaluate_physical_loss()
    model.train()

    cf_thom_ratio = cf_mean_abs / CF_THOM_REFERENCE  # ~1 = right order of magnitude

    # Log as regular step-logged scalars too (not just summary), so they
    # show up in the default charts view / can be added to line panels,
    # not only visible as runs-table columns.
    wandb.log({
        "gap_mass_div_final": gap_mass_div,
        "cf_mean_abs_final": cf_mean_abs,
        "cf_thom_reference": CF_THOM_REFERENCE,
        "cf_thom_ratio_final": cf_thom_ratio,
        "physical_loss_sum_norm_final": norm_sum,
        "physical_loss_sum_raw_final": raw_sum,
        "physical_ns_final": ns_e,
        "physical_data_final": dat_e,
        "physical_bc_final": bc_e,
        "physical_bc_gap_final": bcgap_e,
    })

    wandb.summary["gap_mass_div_final"] = gap_mass_div
    wandb.summary["cf_mean_abs_final"] = cf_mean_abs
    wandb.summary["cf_thom_reference"] = CF_THOM_REFERENCE
    wandb.summary["cf_thom_ratio_final"] = cf_thom_ratio
    wandb.summary["physical_loss_sum_norm_final"] = norm_sum
    wandb.summary["physical_loss_sum_raw_final"] = raw_sum
    wandb.summary["physical_ns_final"] = ns_e
    wandb.summary["physical_data_final"] = dat_e
    wandb.summary["physical_bc_final"] = bc_e
    wandb.summary["physical_bc_gap_final"] = bcgap_e
    # Kept for cross-reference only — NOT the sweep objective, since it
    # requires gap ground truth the PINN is meant to infer without.
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