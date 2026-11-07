import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.integrate import trapezoid
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

# ------------------ CONFIG ------------------
RUN_TAG  = "gD0.080BEST"          # must match RUN_TAG printed by train_pinn_dns.py
SAVE_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Model Saves"
FIG_DIR  = "PINN_Bachelor_Research/Code/FinalThesisCode/Figures"
CMAP       = "viridis"        # F1, F2
RECON_CMAP = "jet"            # F3 (same look as the old per-epoch plot)

# Original Adam-only network: raw grid coordinates, internally normalised.
# DNS and PINN velocities are nondimensional; spatial derivatives are per grid cell.
WALL_EPS = 1e-3
CF_FIT_MAX = 4.0          # radial distance from the physical wall, in cells
CF_BIN_DEG = 12.0         # angular half-width for DNS near-wall regression
CF_MIN_POINTS = 5

os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD ------------------
data = np.load(os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz"))

X, Y = data["X"], data["Y"]
cylinder, gap_mask, ok = data["cylinder"], data["gap_mask"], data["ok"]
cx, cy, r = float(data["cx"]), float(data["cy"]), float(data["r"])
gap_thickness = float(data["gap_thickness"])
u_grid, v_grid = data["u_grid"], data["v_grid"]
H = int(data["h"])
nu = float(data["nu"])
R_WALL = r                  # original network's physical cylinder radius

ny, nx = X.shape
R_outer = r + gap_thickness
D_idx = 2 * r
GD = gap_thickness / D_idx

X_MIN, X_MAX = X.min(), X.max()
Y_MIN, Y_MAX = Y.min(), Y.max()


def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0



class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
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
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


model = PINN().to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt"), map_location=device))
model.eval()
print("Model loaded.")

# ------------------ PREDICTIONS (native grid, raw network output) ------------------
X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
with torch.no_grad():
    u_f, v_f, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
u_pred = u_f.reshape(X.shape).cpu().numpy()
v_pred = v_f.reshape(X.shape).cpu().numpy()

U_true = np.hypot(u_grid, v_grid)
U_pred = np.hypot(u_pred, v_pred)

dist = np.hypot(X - cx, Y - cy)
fluid_eval = (~cylinder) & ok            # metrics only on valid fluid nodes
gap_eval = gap_mask & ok                 # gap measured from the geometric surface
gap_eval_w = gap_eval & (dist > R_WALL)  # gap measured from the effective wall
U_REF = 1.0                              # nondimensional freestream speed
U_INF = U_REF


def rel_l2(pred, true, mask):
    return np.linalg.norm((pred - true)[mask]) / np.linalg.norm(true[mask])


print(f"\n=== g/D = {GD:.3f} (g = {gap_thickness:g} cells), U_inf = {U_INF:.4g}, r_wall = {R_WALL:.2f} ===")
print(f"nu (training) = {nu:g}  ->  effective Re = U_inf * D / nu = {U_INF * D_idx / nu:.3g}")
print(f"inflow-column mean u (should be ~1): {u_grid[:, 0][ok[:, 0]].mean():.4f}")
print(f"excluded from metrics: {(~ok).sum()} NaN nodes ({((~ok) & gap_mask).sum()} in gap)")
for name, p_, t_ in [("u", u_pred, u_grid), ("v", v_pred, v_grid), ("|U|", U_pred, U_true)]:
    print(f"rel L2 {name:>3s}: gap(r) {rel_l2(p_, t_, gap_eval):.4e} | gap(wall) {rel_l2(p_, t_, gap_eval_w):.4e} "
          f"| fluid {rel_l2(p_, t_, fluid_eval):.4e}")
print(f"gap MAE(|U|)/U_ref: {np.mean(np.abs(U_pred - U_true)[gap_eval]) / U_REF:.4e}")

# ------------------ DISPLAY FIELDS ------------------
# No interpolation anywhere: every pixel is a native-grid value (reference) or raw network output (PINN).
# Cells whose centres lie inside the cylinder (or on NaN nodes) can still poke past the exact circle,
# so they show the value of the nearest valid fluid cell. Display only; metrics never use these cells.
_, (_iy, _ix) = distance_transform_edt(~fluid_eval, return_indices=True)
def disp(F):
    return F[_iy, _ix]

uT_d, vT_d, UT_d = disp(u_grid), disp(v_grid), disp(U_true)
uP_d, vP_d, UP_d = disp(u_pred), disp(v_pred), disp(U_pred)

# ------------------ PLOT SETUP ------------------
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

tag = RUN_TAG


def savefig(fig, name):
    # fig.savefig(os.path.join(FIG_DIR, f"{name}_{tag}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}_{tag}.png"))


ext = [(X_MIN - 0.5 - cx) / D_idx, (X_MAX + 0.5 - cx) / D_idx,
       (Y_MIN - 0.5 - cy) / D_idx, (Y_MAX + 0.5 - cy) / D_idx]


def ring_patch(r_in, r_out):
    """Exact annulus (analytic circles, 1440 vertices) used as a clip path in x/D units."""
    t = np.linspace(0, 2 * np.pi, 721)
    outer = np.column_stack([r_out * np.cos(t), r_out * np.sin(t)]) / D_idx
    inner = np.column_stack([r_in * np.cos(t[::-1]), r_in * np.sin(t[::-1])]) / D_idx
    verts = np.vstack([outer, inner])
    codes = np.full(len(verts), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[len(outer)] = Path.MOVETO
    return PathPatch(Path(verts, codes), facecolor="none", edgecolor="none")


R_FAR = 10 * max(nx, ny)     # exterior clip = annulus from r to a radius far beyond the domain


def show(ax, F, lo, hi, cmap, title, gap_only=False):
    # Use fixed levels and common limits so reference/PINN contours are directly comparable.
    levels = np.linspace(lo, hi, 21)
    mask = gap_eval if gap_only else fluid_eval
    field = np.ma.masked_where(~mask, F)
    im = ax.contourf((X - cx) / D_idx, (Y - cy) / D_idx, field,
                     levels=levels, cmap=cmap, vmin=lo, vmax=hi, extend="both",
                     zorder=1)
    clip = ring_patch(r, R_outer if gap_only else R_FAR)
    ax.add_patch(clip)
    im.set_clip_path(clip)                 # pixels are cut by the exact circle, never by pixel squares
    if gap_only:
        ax.add_patch(Circle((0, 0), 0.5, fill=False, edgecolor="k", lw=0.8, zorder=3))
    else:
        ax.add_patch(Circle((0, 0), 0.5, facecolor="0.85", edgecolor="k", lw=0.8, zorder=3))
    
    ax.add_patch(Circle((0, 0), R_outer / D_idx, fill=False, edgecolor="k", lw=0.8, ls="--", zorder=4))
    ax.set_title(title)
    ax.set_xlabel("$x/D$")
    ax.set_ylabel("$y/D$")
    ax.set_aspect("equal")
    return im


# ------------------ F1: full-domain reconstruction (u, v, |U|) ------------------
comps = [("u", u_grid, uT_d, uP_d), ("v", v_grid, vT_d, vP_d), ("|U|", U_true, UT_d, UP_d)]
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), constrained_layout=True)
for i, (name, t_raw, t_d, p_d) in enumerate(comps):
    lo, hi = t_raw[fluid_eval].min(), t_raw[fluid_eval].max()
    e_ = (p_d - t_d) / U_REF
    elim = np.percentile(np.abs(e_[gap_eval]), 99)
    im0 = show(axs[i, 0], t_d, lo, hi, CMAP, f"{name}: reference")
    show(axs[i, 1], p_d, lo, hi, CMAP, f"{name}: PINN")
    im2 = show(axs[i, 2], e_, -elim, elim, "RdBu_r", f"{name}: error / $U_{{ref}}$")
    fig.colorbar(im0, ax=axs[i, :2], shrink=0.8, label=name)
    fig.colorbar(im2, ax=axs[i, 2], shrink=0.8)
fig.suptitle(f"PINN reconstruction, $g/D = {GD:.3f}$ (dashed: gap edge)")
savefig(fig, "F1_reconstruction")

# ------------------ F2: annular-gap zoom ------------------
w = (R_outer + 4) / D_idx
lo, hi = U_true[gap_eval].min(), U_true[gap_eval].max()
e_gap = np.abs(UP_d - UT_d) / U_REF
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
im0 = show(axs[0], UT_d, lo, hi, CMAP, "$|U|$ reference (gap)", gap_only=True)
show(axs[1], UP_d, lo, hi, CMAP, "$|U|$ PINN (gap)", gap_only=True)
im2 = show(axs[2], e_gap, 0, np.percentile(e_gap[gap_eval], 99), "magma",
           r"$||U|_{PINN}-|U|_{ref}|/U_{ref}$", gap_only=True)
for a in axs:
    a.set_xlim(-w, w)
    a.set_ylim(-w, w)
fig.colorbar(im0, ax=axs[:2], shrink=0.8, label="$|U|$")
fig.colorbar(im2, ax=axs[2], shrink=0.8)
savefig(fig, "F2_gap_zoom")

# ------------------ F3: final reconstruction (last state only) ------------------
lo, hi = U_true[fluid_eval].min(), U_true[fluid_eval].max()     # shared limits, as in the original
UH_d = UT_d.copy()
UH_d[gap_mask] = UP_d[gap_mask]                                  # PINN inside gap, ground truth elsewhere

fig, axs = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
im = show(axs[0], UT_d, lo, hi, RECON_CMAP, "Ground truth $|U|$")
show(axs[1], UP_d, lo, hi, RECON_CMAP, "PINN prediction $|U|$")
show(axs[2], UH_d, lo, hi, RECON_CMAP, "PINN in gap + ground truth outside")
fig.colorbar(im, ax=axs, shrink=0.85, label="$|U|$")
savefig(fig, "F3_final_reconstruction")

# ------------------ F4: skin friction, PINN vs DNS ------------------
# C_f = 2 nu / U_inf^2 * (d u_t / dn)_wall.
# DNS: no-slip-constrained least-squares fit on valid native fluid nodes.
# PINN: automatic differentiation at r + WALL_EPS.
# The tangential direction is oriented approximately with the outer flow.
th = np.deg2rad(np.arange(360.0))  # displayed angle: 0 at upstream/front, clockwise
th_phys = (np.pi - th) % (2 * np.pi)  # physical polar angle (CCW from +x)
side = np.where(np.sin(th_phys) >= 0, 1.0, -1.0)


def tt(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device).reshape(-1, 1)


xw = tt(cx + (r + WALL_EPS) * np.cos(th_phys)).requires_grad_(True)
yw = tt(cy + (r + WALL_EPS) * np.sin(th_phys)).requires_grad_(True)
nx_t, ny_t = tt(np.cos(th_phys)), tt(np.sin(th_phys))
tx_t, ty_t = tt(side * np.sin(th_phys)), tt(-side * np.cos(th_phys))
uw, vw, _ = model(xw, yw).split(1, dim=1)
dut_dn = ((grad(uw, xw) * nx_t + grad(uw, yw) * ny_t) * tx_t
          + (grad(vw, xw) * nx_t + grad(vw, yw) * ny_t) * ty_t)
cf_pinn = (2 * nu / U_INF**2 * dut_dn).detach().cpu().numpy().ravel()

ang_nodes = (np.pi - np.arctan2(Y - cy, X - cx)) % (2 * np.pi)
n_nodes = dist - r
fit_nodes = fluid_eval & (n_nodes > 0) & (n_nodes <= CF_FIT_MAX)
cf_dns = np.full(len(th), np.nan)
n_fit = np.zeros(len(th), dtype=int)
for k, t in enumerate(th):
    da = np.angle(np.exp(1j * (ang_nodes - t)))
    mask = fit_nodes & (np.abs(da) <= np.deg2rad(CF_BIN_DEG))
    n = n_nodes[mask]
    if len(n) < CF_MIN_POINTS:
        continue
    t_phys = th_phys[k]
    tx = side[k] * np.sin(t_phys)
    ty = -side[k] * np.cos(t_phys)
    ut = u_grid[mask] * tx + v_grid[mask] * ty
    slope = np.dot(n, ut) / np.dot(n, n)
    cf_dns[k] = 2 * nu * slope / U_INF**2
    n_fit[k] = len(n)

print(f"C_f PINN range: [{np.nanmin(cf_pinn):.4g}, {np.nanmax(cf_pinn):.4g}]")
print(f"C_f DNS fit: {np.isfinite(cf_dns).sum()}/{len(th)} angles; "
      f"samples per valid bin: {n_fit[n_fit > 0].min() if n_fit.any() else 0}–{n_fit.max()}")
print("DNS C_f is a no-slip linear-fit estimate over the first", CF_FIT_MAX, "cells")

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
ax.plot(np.rad2deg(th), cf_dns, color="tab:blue", lw=1.8, label="DNS (near-wall fit)")
ax.plot(np.rad2deg(th), cf_pinn, color="tab:orange", lw=1.5, label="PINN (autodiff)")
ax.axhline(0, color="0.4", lw=0.7)
ax.set(xlabel=r"$\theta$ [deg]", ylabel=r"$C_f$", xlim=(0, 360))
ax.grid(alpha=0.3)
ax.legend()
savefig(fig, "F4a_Cf_theta")

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
cf_difference = cf_pinn - cf_dns
ax.plot(np.rad2deg(th), cf_difference, color="tab:purple", lw=1.5)
ax.axhline(0, color="0.4", lw=0.7)
ax.set(xlabel=r"$\theta$ [deg]", ylabel=r"$C_{f,\mathrm{PINN}}-C_{f,\mathrm{DNS}}$", xlim=(0, 360))
ax.grid(alpha=0.3)
savefig(fig, "F4c_Cf_difference")

fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
lim = max(np.nanmax(np.abs(cf_dns)), np.nanmax(np.abs(cf_pinn)), 1e-12)
for ax, values, name in zip(axs, [cf_dns, cf_pinn], ["DNS (near-wall fit)", "PINN (autodiff)"]):
    sc = ax.scatter(np.cos(th_phys) * r / D_idx, np.sin(th_phys) * r / D_idx,
                    c=values, cmap="RdBu_r", vmin=-lim, vmax=lim, s=25)
    ax.add_patch(Circle((0, 0), r / D_idx, fill=False, edgecolor="k", lw=0.7))
    ax.set(xlabel="$x/D$", ylabel="$y/D$", title=name)
    ax.set_aspect("equal")
fig.colorbar(sc, ax=axs, shrink=0.8, label=r"$C_f$")
savefig(fig, "F4b_Cf_cylinder")

# ------------------ F5: boundary-layer profiles, PINN vs DNS ------------------
# Compare along the same radial rays, with the same tangential velocity projection.
# DNS values are bilinearly interpolated from native nodes. Samples whose
# interpolation stencil contains invalid nodes are omitted (not filled).
angles = np.arange(0, 360, 15)
ETA_MAX, N_ETA = 40.0, 400
SCALE = 18.0

fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax.axhspan(0, gap_thickness / r, color="0.92", zorder=0, label="withheld gap")
valid_float = fluid_eval.astype(float)
bl_profiles = []
for ang in angles:
    t = np.deg2rad(ang)
    t_phys = (np.pi - t) % (2 * np.pi)
    c, sn = np.cos(t_phys), np.sin(t_phys)
    sd = 1.0 if sn >= 0 else -1.0
    tx, ty = sd * sn, -sd * c

    eta = np.linspace(0, ETA_MAX, N_ETA)
    xs, ys = cx + (r + eta) * c, cy + (r + eta) * sn
    inside = ((xs >= X_MIN) & (xs <= X_MAX) &
              (ys >= Y_MIN) & (ys <= Y_MAX))
    eta, xs, ys = eta[inside], xs[inside], ys[inside]
    if len(eta) < 2:
        continue

    with torch.no_grad():
        up, vp, _ = model(tt(xs), tt(ys)).split(1, dim=1)
    ut_p = (up.cpu().numpy().ravel() * tx + vp.cpu().numpy().ravel() * ty) / U_INF

    coords = np.vstack([ys - Y_MIN, xs - X_MIN])
    ud = map_coordinates(u_grid, coords, order=1, mode="nearest")
    vd = map_coordinates(v_grid, coords, order=1, mode="nearest")
    valid = map_coordinates(valid_float, coords, order=1, mode="nearest") > 0.99999
    ut_d = (ud * tx + vd * ty) / U_INF
    ut_d[~valid] = np.nan
    # Both methods share the same no-slip reference value at n=0 for the DNS
    # estimate. The PINN retains its actual wall prediction.
    if np.isclose(eta[0], 0):
        ut_d[0] = 0.0

    # Retain the existing sampled profiles for the thickness calculations below.
    bl_profiles.append((float(ang), eta.copy(), ut_d.copy(), ut_p.copy()))

    nR = eta / r
    ax.plot(ang + SCALE * ut_d, nR, color="tab:blue", lw=1.5,
            label="DNS" if ang == angles[0] else None)
    ax.plot(ang + SCALE * ut_p, nR, color="tab:orange", lw=1.5, ls="--",
            label="PINN" if ang == angles[0] else None)
    ax.axvline(ang, color="0.7", lw=0.6, ls=":")

ax.set_xticks(angles)
ax.set_xticklabels([f"{a}°" for a in angles])
ax.set_xlabel(r"Angle $\theta$ [deg]; horizontal offset proportional to $u_t/U_\infty$")
ax.set_ylabel(r"$n/R$ (from physical wall)")
ax.set_title(r"Tangential boundary-layer velocity profiles, $u_t/U_\infty$")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right")
savefig(fig, "F5_BL_profiles")

# ------------------ F11: boundary-layer thickness comparison ------------------
# The largest tested gap is g/D=0.120 (4.8 cells). DNS profiles inspected there
# approach an outer-flow value in the 20-23.9 cell range at attached sectors. Use
# this same range for every angle; separated sectors may still vary there.
UE_ETA_MIN, UE_ETA_MAX = 20.0, 23.9
THICKNESS_ETA_MAX = 23.9
UE_MIN_ABS = 1e-3 * U_INF


def first_crossing(n, un, threshold):
    """First outward linear crossing; return NaN when no bracket is available."""
    valid = np.isfinite(n) & np.isfinite(un)
    n, un = n[valid], un[valid]
    order = np.argsort(n)
    n, un = n[order], un[order]
    for i in range(len(n) - 1):
        y0, y1 = un[i], un[i + 1]
        if (y0 - threshold) == 0:
            return float(n[i])
        if (y0 - threshold) * (y1 - threshold) <= 0 and y1 != y0:
            return float(n[i] + (threshold - y0) * (n[i + 1] - n[i]) / (y1 - y0))
    return np.nan


def profile_thickness(n, ut, ue):
    if not np.isfinite(ue) or abs(ue) <= UE_MIN_ABS:
        return np.nan, np.nan, np.nan, np.nan
    un = ut / ue
    d50 = first_crossing(n, un, 0.50)
    d90 = first_crossing(n, un, 0.90)
    valid = np.isfinite(n) & np.isfinite(un) & (n >= 0) & (n <= THICKNESS_ETA_MAX)
    n_int, un_int = n[valid], un[valid]
    order = np.argsort(n_int)
    n_int, un_int = n_int[order], un_int[order]
    if len(n_int) < 2 or not np.isclose(n_int[0], 0.0, atol=1e-8):
        return d50, d90, np.nan, np.nan
    delta_star = trapezoid(1.0 - un_int, n_int)
    theta_m = trapezoid(un_int * (1.0 - un_int), n_int)
    return d50, d90, float(delta_star), float(theta_m)


thickness_rows = []
for ang, eta_profile, ut_d_profile, ut_p_profile in bl_profiles:
    # DNS and PINN retain only paired samples, giving identical radial nodes.
    paired = (np.isfinite(eta_profile) & np.isfinite(ut_d_profile) &
              np.isfinite(ut_p_profile) &
              (eta_profile <= THICKNESS_ETA_MAX + ETA_MAX / (N_ETA - 1)))
    n_common = eta_profile[paired]
    dns_common = ut_d_profile[paired]
    pinn_common = ut_p_profile[paired]
    order = np.argsort(n_common)
    n_common, dns_common, pinn_common = n_common[order], dns_common[order], pinn_common[order]

    # The sampled rays share coverage through 23.9 cells; interpolate both
    # profiles to that exact endpoint so every integral has identical limits.
    if len(n_common) >= 2 and n_common[-1] >= THICKNESS_ETA_MAX:
        below_end = n_common < THICKNESS_ETA_MAX
        dns_end = np.interp(THICKNESS_ETA_MAX, n_common, dns_common)
        pinn_end = np.interp(THICKNESS_ETA_MAX, n_common, pinn_common)
        n_common = np.append(n_common[below_end], THICKNESS_ETA_MAX)
        dns_common = np.append(dns_common[below_end], dns_end)
        pinn_common = np.append(pinn_common[below_end], pinn_end)

    edge = ((eta_profile >= UE_ETA_MIN) & (eta_profile <= UE_ETA_MAX) &
            np.isfinite(ut_d_profile))
    ue_values = ut_d_profile[edge]
    Ue = float(np.mean(ue_values)) if len(ue_values) else np.nan
    if not len(n_common) or not np.isclose(n_common[0], 0.0, atol=1e-8):
        print(f"WARNING {ang:5.1f} deg: common profiles do not begin at the wall")
    elif abs(dns_common[0]) > 1e-6 or abs(pinn_common[0]) > 1e-6:
        print(f"WARNING {ang:5.1f} deg: no-slip wall check failed (DNS={dns_common[0]:.3e}, PINN={pinn_common[0]:.3e})")
    if not np.isfinite(Ue) or abs(Ue) <= UE_MIN_ABS:
        print(f"WARNING {ang:5.1f} deg: DNS edge velocity is missing or below {UE_MIN_ABS:.3g}; thicknesses set to NaN")
        vals = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        dns_un = dns_common / Ue
        if np.any(dns_un < -1e-6) or np.any(pinn_common / Ue < -1e-6):
            print(f"WARNING {ang:5.1f} deg: reverse flow is present; thickness values need interpretation")
        dns50, dns90, dnsds, dnstm = profile_thickness(n_common, dns_common, Ue)
        pin50, pin90, pinds, pintm = profile_thickness(n_common, pinn_common, Ue)
        for label, value in [("DNS delta50", dns50), ("PINN delta50", pin50),
                             ("DNS delta90", dns90), ("PINN delta90", pin90)]:
            if not np.isfinite(value):
                print(f"WARNING {ang:5.1f} deg: {label} crossing was not found within the sampled radial domain")
        vals = (dns50, pin50, dns90, pin90, dnsds, pinds, dnstm, pintm)

    thickness_rows.append({
        "theta_deg": ang, "Ue": Ue,
        "DNS_delta50": vals[0], "PINN_delta50": vals[1],
        "DNS_delta90": vals[2], "PINN_delta90": vals[3],
        "DNS_delta_star": vals[4], "PINN_delta_star": vals[5],
        "DNS_theta_m": vals[6], "PINN_theta_m": vals[7],
    })

print(f"\nBoundary-layer thicknesses [grid cells]; Ue interval={UE_ETA_MIN:g}-{UE_ETA_MAX:g}, "
      f"integration interval=0-{THICKNESS_ETA_MAX:g}")
print("theta | Ue | DNS/PINN d50 | DNS/PINN d90 | DNS/PINN d* | DNS/PINN theta_m")
for row in thickness_rows:
    print(f"{row['theta_deg']:5.1f} | {row['Ue']: .4g} | "
          f"{row['DNS_delta50']: .4g}/{row['PINN_delta50']: .4g} | "
          f"{row['DNS_delta90']: .4g}/{row['PINN_delta90']: .4g} | "
          f"{row['DNS_delta_star']: .4g}/{row['PINN_delta_star']: .4g} | "
          f"{row['DNS_theta_m']: .4g}/{row['PINN_theta_m']: .4g}")

import csv
thickness_csv = os.path.join(FIG_DIR, "boundary_layer_thickness.csv")
with open(thickness_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(thickness_rows[0].keys()))
    writer.writeheader()
    writer.writerows(thickness_rows)

thickness_plots = [
    ("F11a_delta50", "DNS_delta50", "PINN_delta50", r"$\delta_{50}$ [cells]", "Boundary-layer $\delta_{50}$"),
    ("F11b_delta90", "DNS_delta90", "PINN_delta90", r"$\delta_{90}$ [cells]", "Boundary-layer $\delta_{90}$"),
    ("F11c_displacement", "DNS_delta_star", "PINN_delta_star", r"$\delta^*$ [cells]", "Displacement thickness"),
    ("F11d_momentum", "DNS_theta_m", "PINN_theta_m", r"$\theta_m$ [cells]", "Momentum thickness"),
]
for name, dns_key, pinn_key, ylabel, title in thickness_plots:
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(angles, [row[dns_key] for row in thickness_rows], color="tab:blue", marker="o", label="DNS")
    ax.plot(angles, [row[pinn_key] for row in thickness_rows], color="tab:orange", marker="s", ls="--", label="PINN")
    ax.set(xlabel=r"Angle $\theta$ [deg]", ylabel=ylabel, title=title, xlim=(0, 360))
    ax.set_xticks(angles)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, f"{name}_{RUN_TAG}.png"), dpi=300)
    # fig.savefig(os.path.join(FIG_DIR, f"{name}_{RUN_TAG}.pdf"))

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
displacement_error = np.array([row["PINN_delta_star"] - row["DNS_delta_star"] for row in thickness_rows])
ax.plot(angles, displacement_error, color="tab:purple", marker="o")
ax.axhline(0, color="0.4", lw=0.7)
ax.set(xlabel=r"Angle $\theta$ [deg]", ylabel=r"PINN $\delta^*$ - DNS $\delta^*$ [cells]",
       title="Displacement-thickness error", xlim=(0, 360))
ax.set_xticks(angles)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG_DIR, f"F11e_displacement_error_{RUN_TAG}.png"), dpi=300)
# fig.savefig(os.path.join(FIG_DIR, f"F11e_displacement_error_{RUN_TAG}.pdf"))

# ------------------ SAVE NUMERICAL COMPARISON ------------------
np.savez(os.path.join(FIG_DIR, f"wall_bl_comparison_{RUN_TAG}.npz"),
         theta_deg=np.rad2deg(th), cf_dns=cf_dns, cf_pinn=cf_pinn,
         cf_dns_fit_count=n_fit)

plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from scipy.ndimage import distance_transform_edt
# from matplotlib.path import Path
# from matplotlib.patches import PathPatch, Circle

# # ------------------ CONFIG ------------------
# RUN_TAG  = "gD0.080"          # must match RUN_TAG printed by train_pinn_dns.py
# SAVE_DIR = "PINN_Bachelor_Research/Code/FinalThesisCode/Model Saves"
# FIG_DIR  = "PINN_Bachelor_Research/Code/FinalThesisCode/Figures"
# CMAP       = "viridis"        # F1, F2
# RECON_CMAP = "jet"            # F3 (same look as the old per-epoch plot)

# # C_f units: derivatives are per GRID CELL in the units the network was trained in (inputs normalised
# # internally, outputs in data units -> nothing to un-normalise). DN_SCALE = 1.0 matches the training
# # loss (nu is in the same units). Only change it for physical units (e.g. 1e3 for mm / (m/s) / SI nu).
# DN_SCALE = 1.0
# WALL_EPS = 1e-3               # [cells] evaluate wall quantities just outside the one-sided hard constraint

# os.makedirs(FIG_DIR, exist_ok=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------ LOAD ------------------
# data = np.load(os.path.join(SAVE_DIR, f"postprocess_data_{RUN_TAG}.npz"))

# X, Y = data["X"], data["Y"]
# cylinder, gap_mask, ok = data["cylinder"], data["gap_mask"], data["ok"]
# cx, cy, r = float(data["cx"]), float(data["cy"]), float(data["r"])
# gap_thickness = float(data["gap_thickness"])
# u_grid, v_grid = data["u_grid"], data["v_grid"]
# H = int(data["h"])
# N_HIDDEN = int(data["n_hidden"])
# nu = float(data["nu"])                       # the SAME nu the NS loss used during training
# R_WALL = float(data["r_wall"])               # effective no-slip radius used in training
# D_LEN = float(data["d_len"])
# HARD_NOSLIP = bool(int(data["hard_noslip"]))
# USE_WALL_FEATURE = bool(int(data["use_wall_feat"])) if "use_wall_feat" in data.files else True

# ny, nx = X.shape
# R_outer = r + gap_thickness
# D_idx = 2 * r
# GD = gap_thickness / D_idx

# X_MIN, X_MAX = X.min(), X.max()
# Y_MIN, Y_MAX = Y.min(), Y.max()


# def normalise(val, vmin, vmax):
#     return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


# class PINN(nn.Module):
#     """Must match the training architecture exactly."""
#     def __init__(self):
#         super().__init__()
#         layers = [nn.Linear(3 if USE_WALL_FEATURE else 2, H), nn.Tanh()]
#         for _ in range(N_HIDDEN - 1):
#             layers += [nn.Linear(H, H), nn.Tanh()]
#         layers += [nn.Linear(H, 3)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, x, y):
#         # x, y are RAW grid coordinates; normalisation happens inside, so autograd derivatives
#         # w.r.t. x, y are already in physical (per-cell) units.
#         xn = normalise(x, X_MIN, X_MAX)
#         yn = normalise(y, Y_MIN, Y_MAX)
#         rho = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12)
#         d = rho - R_WALL
#         feat = torch.tanh(d / (3.0 * D_LEN))
#         inp = torch.cat([xn, yn, feat], dim=1) if USE_WALL_FEATURE else torch.cat([xn, yn], dim=1)
#         out = self.net(inp)
#         u, v, p = out.split(1, dim=1)
#         if HARD_NOSLIP:
#             phi = torch.tanh(d / D_LEN) * (d >= 0).to(out.dtype)
#             u, v = phi * u, phi * v
#         return torch.cat([u, v, p], dim=1)


# def grad(f, x):
#     return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


# model = PINN().to(device)
# model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"cylinder_pinn_{RUN_TAG}.pt"), map_location=device))
# model.eval()
# print("Model loaded.")

# # ------------------ PREDICTIONS (native grid, raw network output) ------------------
# X_flat_t = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
# Y_flat_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
# with torch.no_grad():
#     u_f, v_f, _ = model(X_flat_t, Y_flat_t).split(1, dim=1)
# u_pred = u_f.reshape(X.shape).cpu().numpy()
# v_pred = v_f.reshape(X.shape).cpu().numpy()

# U_true = np.hypot(u_grid, v_grid)
# U_pred = np.hypot(u_pred, v_pred)

# dist = np.hypot(X - cx, Y - cy)
# fluid_eval = (~cylinder) & ok            # metrics only on valid fluid nodes
# gap_eval = gap_mask & ok                 # gap measured from the geometric surface
# gap_eval_w = gap_eval & (dist > R_WALL)  # gap measured from the effective wall
# U_REF = 1.0                              # U_inf of the Re = 100 IBPM data (non-dimensional)
# U_INF = U_REF


# def rel_l2(pred, true, mask):
#     return np.linalg.norm((pred - true)[mask]) / np.linalg.norm(true[mask])


# print(f"\n=== g/D = {GD:.3f} (g = {gap_thickness:g} cells), U_inf = {U_INF:.4g}, r_wall = {R_WALL:.2f} ===")
# print(f"nu (training) = {nu:g}  ->  effective Re = U_inf * D / nu = {U_INF * D_idx / nu:.3g}")
# print(f"inflow-column mean u (should be ~1): {u_grid[:, 0][ok[:, 0]].mean():.4f}")
# print(f"excluded from metrics: {(~ok).sum()} NaN nodes ({((~ok) & gap_mask).sum()} in gap)")
# for name, p_, t_ in [("u", u_pred, u_grid), ("v", v_pred, v_grid), ("|U|", U_pred, U_true)]:
#     print(f"rel L2 {name:>3s}: gap(r) {rel_l2(p_, t_, gap_eval):.4e} | gap(r_wall) {rel_l2(p_, t_, gap_eval_w):.4e} "
#           f"| fluid {rel_l2(p_, t_, fluid_eval):.4e}")
# print(f"gap MAE(|U|)/U_ref: {np.mean(np.abs(U_pred - U_true)[gap_eval]) / U_REF:.4e}")

# # ------------------ DISPLAY FIELDS ------------------
# # No interpolation anywhere: every pixel is a native-grid value (reference) or raw network output (PINN).
# # Cells whose centres lie inside the cylinder (or on NaN nodes) can still poke past the exact circle,
# # so they show the value of the nearest valid fluid cell. Display only; metrics never use these cells.
# _, (_iy, _ix) = distance_transform_edt(~fluid_eval, return_indices=True)
# def disp(F):
#     return F[_iy, _ix]

# uT_d, vT_d, UT_d = disp(u_grid), disp(v_grid), disp(U_true)
# uP_d, vP_d, UP_d = disp(u_pred), disp(v_pred), disp(U_pred)

# # ------------------ PLOT SETUP ------------------
# plt.rcParams.update({
#     "font.family": "serif", "mathtext.fontset": "cm",
#     "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
#     "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
# })

# tag = RUN_TAG


# def savefig(fig, name):
#     fig.savefig(os.path.join(FIG_DIR, f"{name}_{tag}.pdf"))
#     fig.savefig(os.path.join(FIG_DIR, f"{name}_{tag}.png"))


# ext = [(X_MIN - 0.5 - cx) / D_idx, (X_MAX + 0.5 - cx) / D_idx,
#        (Y_MIN - 0.5 - cy) / D_idx, (Y_MAX + 0.5 - cy) / D_idx]


# def ring_patch(r_in, r_out):
#     """Exact annulus (analytic circles, 1440 vertices) used as a clip path in x/D units."""
#     t = np.linspace(0, 2 * np.pi, 721)
#     outer = np.column_stack([r_out * np.cos(t), r_out * np.sin(t)]) / D_idx
#     inner = np.column_stack([r_in * np.cos(t[::-1]), r_in * np.sin(t[::-1])]) / D_idx
#     verts = np.vstack([outer, inner])
#     codes = np.full(len(verts), Path.LINETO)
#     codes[0] = Path.MOVETO
#     codes[len(outer)] = Path.MOVETO
#     return PathPatch(Path(verts, codes), facecolor="none", edgecolor="none")


# R_FAR = 10 * max(nx, ny)     # exterior clip = annulus from r to a radius far beyond the domain


# def show(ax, F, lo, hi, cmap, title, gap_only=False):
#     im = ax.imshow(F, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi,
#                    interpolation="nearest", zorder=1)
#     clip = ring_patch(r, R_outer if gap_only else R_FAR)
#     ax.add_patch(clip)
#     im.set_clip_path(clip)                 # pixels are cut by the exact circle, never by pixel squares
#     if gap_only:
#         ax.add_patch(Circle((0, 0), 0.5, fill=False, edgecolor="k", lw=0.8, zorder=3))
#     else:
#         ax.add_patch(Circle((0, 0), 0.5, facecolor="0.85", edgecolor="k", lw=0.8, zorder=3))
#     ax.add_patch(Circle((0, 0), R_WALL / D_idx, fill=False, edgecolor="k", lw=0.6, ls=":", zorder=4))   # effective wall
#     ax.add_patch(Circle((0, 0), R_outer / D_idx, fill=False, edgecolor="k", lw=0.8, ls="--", zorder=4))
#     ax.set_title(title)
#     ax.set_xlabel("$x/D$")
#     ax.set_ylabel("$y/D$")
#     ax.set_aspect("equal")
#     return im


# # ------------------ F1: full-domain reconstruction (u, v, |U|) ------------------
# comps = [("u", u_grid, uT_d, uP_d), ("v", v_grid, vT_d, vP_d), ("|U|", U_true, UT_d, UP_d)]
# fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), constrained_layout=True)
# for i, (name, t_raw, t_d, p_d) in enumerate(comps):
#     lo, hi = t_raw[fluid_eval].min(), t_raw[fluid_eval].max()
#     e_ = (p_d - t_d) / U_REF
#     elim = np.percentile(np.abs(e_[gap_eval]), 99)
#     im0 = show(axs[i, 0], t_d, lo, hi, CMAP, f"{name}: reference")
#     show(axs[i, 1], p_d, lo, hi, CMAP, f"{name}: PINN")
#     im2 = show(axs[i, 2], e_, -elim, elim, "RdBu_r", f"{name}: error / $U_{{ref}}$")
#     fig.colorbar(im0, ax=axs[i, :2], shrink=0.8, label=name)
#     fig.colorbar(im2, ax=axs[i, 2], shrink=0.8)
# fig.suptitle(f"PINN reconstruction, $g/D = {GD:.3f}$ (dashed: gap edge, dotted: effective wall)")
# savefig(fig, "F1_reconstruction")

# # ------------------ F2: annular-gap zoom ------------------
# w = (R_outer + 4) / D_idx
# lo, hi = U_true[gap_eval].min(), U_true[gap_eval].max()
# e_gap = np.abs(UP_d - UT_d) / U_REF
# fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
# im0 = show(axs[0], UT_d, lo, hi, CMAP, "$|U|$ reference (gap)", gap_only=True)
# show(axs[1], UP_d, lo, hi, CMAP, "$|U|$ PINN (gap)", gap_only=True)
# im2 = show(axs[2], e_gap, 0, np.percentile(e_gap[gap_eval], 99), "magma",
#            r"$||U|_{PINN}-|U|_{ref}|/U_{ref}$", gap_only=True)
# for a in axs:
#     a.set_xlim(-w, w)
#     a.set_ylim(-w, w)
# fig.colorbar(im0, ax=axs[:2], shrink=0.8, label="$|U|$")
# fig.colorbar(im2, ax=axs[2], shrink=0.8)
# savefig(fig, "F2_gap_zoom")

# # ------------------ F3: final reconstruction (last state only) ------------------
# lo, hi = U_true[fluid_eval].min(), U_true[fluid_eval].max()     # shared limits, as in the original
# UH_d = UT_d.copy()
# UH_d[gap_mask] = UP_d[gap_mask]                                  # PINN inside gap, ground truth elsewhere

# fig, axs = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
# im = show(axs[0], UT_d, lo, hi, RECON_CMAP, "Ground truth $|U|$")
# show(axs[1], UP_d, lo, hi, RECON_CMAP, "PINN prediction $|U|$")
# show(axs[2], UH_d, lo, hi, RECON_CMAP, "PINN in gap + ground truth outside")
# fig.colorbar(im, ax=axs, shrink=0.85, label="$|U|$")
# savefig(fig, "F3_final_reconstruction")

# # ------------------ F4: skin friction coefficient on the (effective) wall ------------------
# # C_f = 2 tau_w / (rho U^2) = 2 nu (d u_t / d n)|_wall / U_inf^2   (rho cancels)
# # Wall = effective no-slip radius R_WALL (what the network was trained with).
# # n = outward radial normal; tangent t = side * (sin th, -cos th), oriented WITH the flow
# # (flow assumed +x, front stagnation point at th = 180 deg): attached flow -> C_f > 0, separation -> sign change.
# th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
# side = np.where(np.sin(th) > 0, 1.0, -1.0)


# def tt(a):
#     return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(1)


# r_eval = R_WALL + WALL_EPS
# xw = tt(cx + r_eval * np.cos(th)).requires_grad_(True)
# yw = tt(cy + r_eval * np.sin(th)).requires_grad_(True)
# n_x, n_y, sd = tt(np.cos(th)), tt(np.sin(th)), tt(side)

# uw, vw, _ = model(xw, yw).split(1, dim=1)
# dudn = grad(uw, xw) * n_x + grad(uw, yw) * n_y
# dvdn = grad(vw, xw) * n_x + grad(vw, yw) * n_y
# dut_dn = sd * (dudn * n_y - dvdn * n_x) * DN_SCALE

# Cf = (2.0 * nu * dut_dn / U_INF ** 2).detach().cpu().numpy().ravel()
# print(f"C_f range: [{Cf.min():.3e}, {Cf.max():.3e}]")

# fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
# ax.plot(np.degrees(th), Cf, "k-", lw=1.2, label="PINN (autodiff at effective wall)")

# # # Reference wall-shear proxy from native nodes (no interpolation): no-slip-constrained fit of u_t vs distance
# # # from the effective wall, using nodes between 1 cell beyond the wall and the gap edge.
# # th_g = np.arctan2(Y - cy, X - cx) % (2 * np.pi)
# # n_g = dist - R_WALL
# # sel = (n_g >= 1.0) & (n_g <= (R_outer - R_WALL)) & ok & (~cylinder)
# # ut_g = np.where(np.sin(th_g) > 0, 1.0, -1.0) * (u_grid * np.sin(th_g) - v_grid * np.cos(th_g))
# # bins = np.linspace(0, 2 * np.pi, 61)
# # th_c = 0.5 * (bins[1:] + bins[:-1])
# # cf_ref = np.full(60, np.nan)
# # for k in range(60):
# #     m = sel & (th_g >= bins[k]) & (th_g < bins[k + 1])
# #     if m.sum() >= 3:
# #         cf_ref[k] = 2 * nu * np.sum(n_g[m] * ut_g[m]) / np.sum(n_g[m] ** 2) / U_INF ** 2
# # ax.plot(np.degrees(th_c), cf_ref, "ro", ms=3, label="reference (fit on native nodes)")

# ax.axhline(0, color="0.5", lw=0.6)
# ax.set_xlabel(r"$\theta$ [deg]")
# ax.set_ylabel("$C_f$")
# ax.set_xlim(0, 360)
# ax.legend()
# ax.grid(alpha=0.3)
# savefig(fig, "F4a_Cf_theta")

# fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
# m_ = np.max(np.abs(Cf))
# sc = ax.scatter(R_WALL * np.cos(th) / D_idx, R_WALL * np.sin(th) / D_idx, c=Cf, cmap="seismic", vmin=-m_, vmax=m_, s=30)
# ax.add_patch(Circle((0, 0), R_WALL / D_idx, fill=False, edgecolor="k", lw=0.8))
# ax.set_aspect("equal")
# ax.set_xlabel("$x/D$")
# ax.set_ylabel("$y/D$")
# fig.colorbar(sc, ax=ax, shrink=0.8, label="$C_f$")
# savefig(fig, "F4b_Cf_cylinder")

# # ------------------ F5: boundary-layer profiles ------------------
# # u_t / U_inf along the outward normal at several angles, n measured from the effective wall.
# # Profile drawn at x = theta + SCALE * u_t/U_inf. Rays are truncated at the edge of the cropped domain
# # (the network was never trained outside it).
# angles = np.arange(0, 360, 30)
# ETA_MAX, N_ETA = 40.0, 200
# SCALE = 18.0

# fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
# colors = plt.cm.turbo(np.linspace(0, 1, len(angles)))
# ax.axhspan(0, (R_outer - R_WALL) / r, color="0.9", zorder=0, label="gap")

# for i, ang in enumerate(angles):
#     t_ = np.radians(ang)
#     c_, s_ = np.cos(t_), np.sin(t_)
#     sd_ = 1.0 if s_ > 1e-9 else -1.0
#     t_x, t_y = sd_ * s_, -sd_ * c_

#     eta = np.linspace(0, ETA_MAX, N_ETA)
#     xs, ys = cx + (R_WALL + eta) * c_, cy + (R_WALL + eta) * s_
#     inside = (xs >= X_MIN) & (xs <= X_MAX) & (ys >= Y_MIN) & (ys <= Y_MAX)
#     eta, xs, ys = eta[inside], xs[inside], ys[inside]
#     if len(eta) < 2:
#         continue

#     with torch.no_grad():
#         u, v, _ = model(tt(xs), tt(ys)).split(1, dim=1)
#     ut = (u.cpu().numpy().ravel() * t_x + v.cpu().numpy().ravel() * t_y) / U_INF
#     nR = eta / r
#     x_plot = ang + ut * SCALE

#     ax.fill_betweenx(nR, ang, x_plot, alpha=0.25, color=colors[i])
#     ax.plot(x_plot, nR, color=colors[i], lw=1.5, label=f"{ang}°")
#     ax.axvline(ang, color=colors[i], lw=0.7, ls="--", alpha=0.5)

# ax.set_xticks(angles)
# ax.set_xticklabels([f"{a}°" for a in angles])
# ax.set_xlabel(r"Angle $\theta$ [deg]")
# ax.set_ylabel("$n / R$ (from effective wall)")
# ax.set_title("Boundary layer profiles $u_t/U_\\infty$")
# ax.legend(loc="upper right", fontsize=8, ncol=3)
# ax.grid(True, axis="y", ls="--", alpha=0.4)
# ax.set_ylim(bottom=0)
# savefig(fig, "F5_BL_profiles")

# plt.show()

# wz = np.gradient(v_grid, axis=1) - np.gradient(u_grid, axis=0)
# wzx, wzy = np.gradient(wz, axis=1), np.gradient(wz, axis=0)
# res = u_grid*wzx + v_grid*wzy - nu*(np.gradient(wzx, axis=1) + np.gradient(wzy, axis=0))
# conv = np.abs(u_grid*wzx + v_grid*wzy)
# for name, m in [("gap", gap_eval & (dist > R_WALL)), ("wake", (~cylinder) & (X > cx + r + 10) & (np.abs(Y - cy) < 15))]:
#     print(name, "rms residual / rms convection:", np.sqrt(np.mean(res[m]**2)) / np.sqrt(np.mean(conv[m]**2)))

# def vort_res(u, v):
#     wz = np.gradient(v, axis=1) - np.gradient(u, axis=0)
#     wzx, wzy = np.gradient(wz, axis=1), np.gradient(wz, axis=0)
#     conv = u * wzx + v * wzy
#     return conv - nu * (np.gradient(wzx, axis=1) + np.gradient(wzy, axis=0)), conv

# n_w = dist - r
# print("wall dist [cells] | DNS res/conv | PINN res/conv")
# for lo, hi in [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5), (5, 8), (8, 12)]:
#     m = (n_w >= lo) & (n_w < hi) & ok
#     row = []
#     for uu, vv in [(u_grid, v_grid), (u_pred, v_pred)]:
#         res, conv = vort_res(uu, vv)
#         row.append(np.sqrt(np.mean(res[m] ** 2)) / np.sqrt(np.mean(conv[m] ** 2)))
#     print(f"[{lo:>4}, {hi:>4})        | {row[0]:10.2f}   | {row[1]:10.2f}")
