import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from matplotlib.patches import Circle

# ------------------ CONFIG ------------------
# IMPORTANT: this must exactly match RUN_TAG in the training script.
RUN_TAG = "gD0.080"
SAVE_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Model Saves"
FIG_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Figures"
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------ LOAD SAVED DATA ------------------
data = np.load(os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz"))

X, Y = data["X"], data["Y"]
cylinder = data["cylinder"].astype(bool)
gap_mask = data["gap_mask"].astype(bool)
ok = data["ok"].astype(bool)
cx, cy, r = float(data["cx"]), float(data["cy"]), float(data["r"])
gap_thickness = float(data["gap_thickness"])
u_grid, v_grid = data["u_grid"], data["v_grid"]
h, nu = int(data["h"]), float(data["nu"])

ny, nx = X.shape
R_outer = r + gap_thickness
D_idx = 2.0 * r
GD = gap_thickness / D_idx
U_REF = 1.0
X_MIN, X_MAX = float(X.min()), float(X.max())
Y_MIN, Y_MAX = float(Y.min()), float(Y.max())

dist = np.hypot(X - cx, Y - cy)
fluid_eval = (~cylinder) & ok
gap_eval = gap_mask & ok
U_true = np.hypot(u_grid, v_grid)

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
        return torch.cat([phi*u_raw, phi*v_raw, p], dim=1)

model = PINN().to(device)
model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt"),
    map_location=device
))
model.eval()
print("Model loaded.")

def tt(a, requires_grad=False):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(requires_grad)

# ------------------ GRID PREDICTIONS ------------------
with torch.no_grad():
    pred = model(tt(X.ravel()), tt(Y.ravel()))
u_pred = pred[:, 0].cpu().numpy().reshape(X.shape)
v_pred = pred[:, 1].cpu().numpy().reshape(X.shape)
U_pred = np.hypot(u_pred, v_pred)

def rel_l2(prediction, reference, mask):
    den = np.linalg.norm(reference[mask])
    return np.linalg.norm((prediction-reference)[mask]) / den if den > 0 else np.nan

vec_error = np.hypot(u_pred-u_grid, v_pred-v_grid)
u_error = u_pred-u_grid
v_error = v_pred-v_grid
speed_error = U_pred-U_true

print(f"\nRun: {RUN_TAG} | g/D={GD:.3f} | Re={1.0*D_idx/nu:.3f}")
for name, p, t in [
    ("u", u_pred, u_grid), ("v", v_pred, v_grid),
    ("speed", U_pred, U_true), ("vector", vec_error, np.zeros_like(vec_error))
]:
    if name == "vector":
        value = np.linalg.norm(vec_error[gap_eval]) / np.linalg.norm(
            np.hypot(u_grid[gap_eval], v_grid[gap_eval]))
    else:
        value = rel_l2(p, t, gap_eval)
    print(f"Gap relative L2 {name:>6s}: {value:.6e}")
print(f"Gap mean absolute vector error / Uref: {np.mean(vec_error[gap_eval]):.6e}")

# ------------------ POINTWISE NS RESIDUALS ------------------
def NS_residual_components(x, y):
    u, v, p = model(x, y).split(1, dim=1)

    def derivative(f, z, create_graph):
        return torch.autograd.grad(
            f, z, grad_outputs=torch.ones_like(f),
            create_graph=create_graph, retain_graph=True
        )[0]

    ux = derivative(u, x, True)
    uy = derivative(u, y, True)
    vx = derivative(v, x, True)
    vy = derivative(v, y, True)
    uxx = derivative(ux, x, False)
    uyy = derivative(uy, y, False)
    vxx = derivative(vx, x, False)
    vyy = derivative(vy, y, False)
    px = derivative(p, x, False)
    py = derivative(p, y, False)

    cont = ux + vy
    momx = u*ux + v*uy - nu*(uxx+uyy) + px
    momy = u*vx + v*vy - nu*(vxx+vyy) + py
    return cont.detach(), momx.detach(), momy.detach()

fluid_geom = ~cylinder
coords = np.column_stack((X[fluid_geom], Y[fluid_geom])).astype(np.float32)
res_cont = np.full(X.shape, np.nan)
res_momx = np.full(X.shape, np.nan)
res_momy = np.full(X.shape, np.nan)

chunk_size = 512
for start in range(0, len(coords), chunk_size):
    xy = coords[start:start+chunk_size]
    xt = tt(xy[:, 0], True)
    yt = tt(xy[:, 1], True)
    with torch.enable_grad():
        rc, rx, ry = NS_residual_components(xt, yt)
    iy = np.rint(xy[:, 1]).astype(int)
    ix = np.rint(xy[:, 0]).astype(int)
    res_cont[iy, ix] = rc.cpu().numpy().ravel()
    res_momx[iy, ix] = rx.cpu().numpy().ravel()
    res_momy[iy, ix] = ry.cpu().numpy().ravel()

res_total = np.sqrt(res_cont**2 + res_momx**2 + res_momy**2)
dns_error = vec_error.copy()
dns_error[~fluid_eval] = np.nan

# ------------------ SUMMARY STATISTICS ------------------
def residual_stats(label, values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        print(f"{label}: no valid points")
        return
    print(f"{label:24s} MAE={np.mean(np.abs(values)):.6e} "
          f"RMS={np.sqrt(np.mean(values**2)):.6e} "
          f"max|.|={np.max(np.abs(values)):.6e}")

print("\nGap residual statistics:")
for name, arr in [
    ("Continuity", res_cont), ("X momentum", res_momx),
    ("Y momentum", res_momy), ("Combined", res_total)
]:
    residual_stats(name, arr[gap_eval])

# Radial bands: half-cell wide across the first four cells from the physical wall.
print("\nRadial-band reconstruction and residual statistics:")
radial_rows = []
radial_limit = min(4.0, gap_thickness)
radial_edges = np.arange(0.0, radial_limit, 0.5)
if radial_limit > 0 and (not len(radial_edges) or radial_edges[-1] < radial_limit):
    radial_edges = np.append(radial_edges, radial_limit)
for inner, outer in zip(radial_edges[:-1], radial_edges[1:]):
    mask = gap_eval & (dist > r+inner) & (dist <= r+outer+1e-9)
    if not np.any(mask):
        continue
    ref_norm = np.linalg.norm(np.hypot(u_grid[mask], v_grid[mask]))
    row = {
        "band_inner_cells": float(inner),
        "band_outer_cells": float(outer),
        "n": int(mask.sum()),
        "rel_l2_vector": float(np.linalg.norm(vec_error[mask]) / ref_norm) if ref_norm > 0 else np.nan,
        "mean_abs_vector_error": float(np.mean(vec_error[mask])),
        "rms_vector_error": float(np.sqrt(np.mean(vec_error[mask]**2))),
        "rel_l2_u": rel_l2(u_pred, u_grid, mask),
        "rel_l2_v": rel_l2(v_pred, v_grid, mask),
        "rel_l2_speed": rel_l2(U_pred, U_true, mask),
        "mean_vector_error": float(np.mean(vec_error[mask])),
        "rms_vector_error": float(np.sqrt(np.mean(vec_error[mask]**2))),
        "rms_cont": float(np.sqrt(np.mean(res_cont[mask]**2))),
        "rms_momx": float(np.sqrt(np.mean(res_momx[mask]**2))),
        "rms_momy": float(np.sqrt(np.mean(res_momy[mask]**2))),
    }
    radial_rows.append(row)
    print(row)

# Angular sectors: 12 sectors, 30 degrees each.
theta = (np.degrees(np.arctan2(Y-cy, X-cx)) + 360.0) % 360.0
print("\nAngular-sector reconstruction and residual statistics:")
angular_rows = []
for a0 in range(0, 360, 30):
    a1 = a0 + 30
    mask = gap_eval & (theta >= a0) & (theta < a1)
    if not np.any(mask):
        continue
    ref_norm = np.linalg.norm(np.hypot(u_grid[mask], v_grid[mask]))
    row = {
        "theta_start_deg": a0,
        "theta_end_deg": a1,
        "n": int(mask.sum()),
        "rel_l2_vector": float(np.linalg.norm(vec_error[mask]) / ref_norm) if ref_norm > 0 else np.nan,
        "mean_vector_error": float(np.mean(vec_error[mask])),
        "rms_cont": float(np.sqrt(np.mean(res_cont[mask]**2))),
        "rms_momx": float(np.sqrt(np.mean(res_momx[mask]**2))),
        "rms_momy": float(np.sqrt(np.mean(res_momy[mask]**2))),
    }
    angular_rows.append(row)
    print(row)

# ------------------ SAVE NUMERICAL RESULTS ------------------
np.savez(
    os.path.join(FIG_DIR, f"diagnostics_{RUN_TAG}.npz"),
    u_pred=u_pred, v_pred=v_pred, U_pred=U_pred,
    u_error=u_error, v_error=v_error, vec_error=vec_error,
    res_cont=res_cont, res_momx=res_momx, res_momy=res_momy,
    res_total=res_total, dns_vector_error=dns_error,
    radial_rows=np.array(radial_rows, dtype=object),
    angular_rows=np.array(angular_rows, dtype=object)
)

# ------------------ FIGURE F6: COMPONENT AND VECTOR ERROR MAPS ------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
error_fields = [
    (u_error/U_REF, "u error / Uref"),
    (v_error/U_REF, "v error / Uref"),
    (vec_error, "velocity-vector error")
]
for ax, (field, title) in zip(axs, error_fields):
    field_plot = np.where(fluid_eval, field, np.nan)
    lim = np.nanpercentile(np.abs(field_plot), 99)
    im = ax.imshow(field_plot, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                   cmap="RdBu_r" if title != "velocity-vector error" else "magma",
                   vmin=-lim if title != "velocity-vector error" else 0,
                   vmax=lim)
    ax.add_patch(Circle((cx, cy), r, fill=False, color="black", lw=1))
    ax.add_patch(Circle((cx, cy), R_outer, fill=False, color="white", lw=1, ls="--"))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
fig.savefig(os.path.join(FIG_DIR, f"F6_velocity_error_{RUN_TAG}.png"), dpi=300)
# fig.savefig(os.path.join(FIG_DIR, f"F6_velocity_error_{RUN_TAG}.pdf"))

# ------------------ FIGURE F7: RADIAL-BAND ERROR ------------------
if radial_rows:
    radial_x = [0.5*(x["band_inner_cells"] + x["band_outer_cells"]) for x in radial_rows]
    for suffix, key, ylabel, title in [
        ("rel_l2", "rel_l2_vector", "Relative L2 vector error", "Near-wall relative L2 vector error"),
        ("mae", "mean_abs_vector_error", "Mean absolute vector error", "Near-wall mean absolute vector error")
    ]:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(radial_x, [x[key] for x in radial_rows], marker="o")
        ax.set_xlabel("Radial distance from cylinder wall [cells]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(FIG_DIR, f"F7_radial_{suffix}_{RUN_TAG}.png"), dpi=300)
        # fig.savefig(os.path.join(FIG_DIR, f"F7_radial_{suffix}_{RUN_TAG}.pdf"))

# ------------------ FIGURE F8: ANGULAR-SECTOR ERROR ------------------
if angular_rows:
    sector_labels = [f"{x['theta_start_deg']}°" for x in angular_rows]
    vals = [x["rel_l2_vector"] for x in angular_rows]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(sector_labels, vals, width=0.8)
    ax.set_xlabel("Angular sector start")
    ax.set_ylabel("Relative L2 vector error")
    ax.set_title("Near-wall reconstruction error by angular sector")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, f"F8_angular_error_{RUN_TAG}.png"), dpi=300)
    # fig.savefig(os.path.join(FIG_DIR, f"F8_angular_error_{RUN_TAG}.pdf"))

# ------------------ FIGURE F9: NS RESIDUAL MAPS ------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
for ax, arr, title in zip(
    axs.flat,
    [res_cont, res_momx, res_momy, res_total],
    ["Continuity residual", "X-momentum residual", "Y-momentum residual", "Combined residual"]
):
    field = np.where(fluid_geom, np.log10(np.maximum(np.abs(arr), 1e-12)), np.nan)
    im = ax.imshow(field, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                   cmap="viridis", aspect="equal")
    ax.add_patch(Circle((cx, cy), r, fill=False, color="white", lw=1))
    ax.add_patch(Circle((cx, cy), R_outer, fill=False, color="red", lw=1, ls="--"))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="log10 absolute residual")
fig.savefig(os.path.join(FIG_DIR, f"F9_NS_residuals_{RUN_TAG}.png"), dpi=300)
# fig.savefig(os.path.join(FIG_DIR, f"F9_NS_residuals_{RUN_TAG}.pdf"))

# ------------------ FIGURE F10: NEAR-WALL ERROR DISTRIBUTION ------------------

near_wall = gap_eval

fields = [
    (vec_error, "Velocity-vector error"),
    (np.abs(res_cont), "Continuity residual"),
    (np.abs(res_momx), "X-momentum residual"),
    (np.abs(res_momy), "Y-momentum residual")
]

fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

for ax, (field, title) in zip(axs.flat, fields):

    values = np.where(near_wall, field, np.nan)

    im = ax.imshow(
        values,
        origin="lower",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="magma",
        interpolation="nearest"
    )

    ax.add_patch(Circle((cx, cy), r, fill=False,
                        color="white", lw=1))
    ax.add_patch(Circle((cx, cy), R_outer, fill=False,
                        color="cyan", lw=1, ls="--"))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)

fig.savefig(os.path.join(FIG_DIR, f"F10_near_wall_diagnostics_{RUN_TAG}.png"), dpi=300)
# fig.savefig(os.path.join(FIG_DIR, f"F10_near_wall_diagnostics_{RUN_TAG}.pdf"))

# ------------------ SAVE RADIAL / ANGULAR TABLES ------------------
import csv
for filename, rows in [
    (f"radial_stats_{RUN_TAG}.csv", radial_rows),
    (f"angular_stats_{RUN_TAG}.csv", angular_rows)
]:
    if rows:
        with open(os.path.join(FIG_DIR, filename), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

print(f"\nDiagnostics saved in: {FIG_DIR}")
# plt.show()
print(next(model.parameters()).device)
print(torch.cuda.is_available())
