import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load("PINN_Bachelor_Research/Code/RealCyltFlow/Model Saves/cylinder_pinn_imp0.02.pt")

X = data["X"]
Y = data["Y"]
cylinder = data["cylinder"]
gap_mask = data["gap_mask"]

cx = float(data["cx"])
cy = float(data["cy"])
r  = float(data["r"])

u_grid = data["u_grid"]
v_grid = data["v_grid"]

gap_thickness = float(data["gap_thickness"])

X_MIN = X.min()
X_MAX = X.max()

Y_MIN = Y.min()
Y_MAX = Y.max()

h = 64 #128

def normalise(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h),
            nn.Tanh(),

            nn.Linear(h, h),
            nn.Tanh(),

            nn.Linear(h, h),
            nn.Tanh(),

            # nn.Linear(h, h),
            # nn.Tanh(),

            # nn.Linear(h, h),
            # nn.Tanh(),

            nn.Linear(h, 3)
        )

    def forward(self, x, y):

        xn = normalise(x, X_MIN, X_MAX)
        yn = normalise(y, Y_MIN, Y_MAX)

        return self.net(torch.cat([xn, yn], dim=1))


def grad(f, x):
    return torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

model = PINN().to(device)

model.load_state_dict(
    torch.load(
        "PINN_Bachelor_Research/Code/RealCyltFlow/Model Saves/cylinder_pinn_imp0.02long.pt",
        map_location=device
    )
)

model.eval()

print("Model loaded.")

X_flat_t = torch.tensor(
    X.flatten(),
    dtype=torch.float32,
    device=device
).unsqueeze(1)

Y_flat_t = torch.tensor(
    Y.flatten(),
    dtype=torch.float32,
    device=device
).unsqueeze(1)

with torch.no_grad():
    u_full, v_full, p_full = model(X_flat_t, Y_flat_t).split(1, dim=1)

u_full = u_full.reshape(X.shape).cpu().numpy()
v_full = v_full.reshape(X.shape).cpu().numpy()

U_full = np.sqrt(u_full**2 + v_full**2)

valid_far = (~cylinder) & (~np.isnan(U_full))

U_inf = 5 #np.percentile(U_full[valid_far], 95)

print("U_inf: ", U_inf)

rho = 1.225
nu = 1.5e-5 #mu is nu*rho

theta = np.linspace(0, 2*np.pi, 360)

x_wall = cx + r*np.cos(theta)
y_wall = cy + r*np.sin(theta)

xw_t = torch.tensor(
    x_wall,
    dtype=torch.float32,
    device=device
).unsqueeze(1)

yw_t = torch.tensor(
    y_wall,
    dtype=torch.float32,
    device=device
).unsqueeze(1)

xw_t.requires_grad_(True)
yw_t.requires_grad_(True)

uw, vw, _ = model(xw_t, yw_t).split(1, dim=1)

ux = grad(uw, xw_t)
uy = grad(uw, yw_t)

vx = grad(vw, xw_t)
vy = grad(vw, yw_t)

nx = torch.cos(
    torch.tensor(theta, dtype=torch.float32, device=device)
).unsqueeze(1)

ny = torch.sin(
    torch.tensor(theta, dtype=torch.float32, device=device)
).unsqueeze(1)

tx = -ny
ty = nx

du_dn = (ux*nx + uy*ny)*1e3
dv_dn = (vx*nx + vy*ny)*1e3

dut_dn = du_dn*tx + dv_dn*ty

tau_w = rho * nu * dut_dn

Cf = 2 * tau_w / (rho * U_inf**2)

Cf_n = np.abs(Cf.detach().cpu().numpy().flatten())

plt.figure(figsize=(8,4))

plt.plot(np.degrees(theta), Cf_n)

plt.xlabel("theta [deg]")
plt.ylabel("C_f")

plt.title("Skin Friction Coefficient")

plt.grid(True)

plt.show()

angles  = np.arange(0, 361, 30)
eta_max = 40
N_eta   = 200
U_inf   = 5.0
scale   = 18.0   # horizontal scale: how many degrees wide one full U_inf unit spans
 
fig, ax = plt.subplots(figsize=(14, 5))
 
colors = plt.cm.tab10(np.linspace(0, 1, len(angles)))
 
for i, ang in enumerate(angles):
 
    th     = np.radians(ang)
    x0     = cx + r * np.cos(th)
    y0     = cy + r * np.sin(th)
 
    nx_vec = np.cos(th)
    ny_vec = np.sin(th)

    tx = -ny_vec
    ty =  nx_vec

    # flip tangent so it always points in the +x half-plane
    # i.e. aligns with the free-stream direction locally
    if tx < 0:
        tx = -tx
        ty = -ty
 
    eta = np.linspace(0, eta_max, N_eta)
    xs  = x0 + nx_vec * eta
    ys  = y0 + ny_vec * eta
 
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device).unsqueeze(1)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
 
    with torch.no_grad():
        u, v, _ = model(xs_t, ys_t).split(1, dim=1)
 
    u  = u.cpu().numpy().flatten()
    v  = v.cpu().numpy().flatten()
    ut = (u * tx + v * ty) / U_inf
    nR = eta / r
 
    # x-position = angle + scaled ut value
    x_plot = ang + ut * scale
 
    # filled area between zero-line and profile
    ax.fill_betweenx(
        nR,
        ang,           # zero line at the angle position
        x_plot,
        alpha=0.25,
        color=colors[i]
    )
 
    # profile line
    ax.plot(
        x_plot,
        nR,
        color=colors[i],
        linewidth=1.5,
        label=f'{ang}°'
    )
 
    # zero reference line
    ax.axvline(
        ang,
        color=colors[i],
        linewidth=0.7,
        linestyle='--',
        alpha=0.5
    )
 
# ---- formatting ----
ax.set_xticks(angles)
ax.set_xticklabels([f'{a}°' for a in angles])
ax.set_xlabel('Angle  θ  [deg]')
ax.set_ylabel('n / R')
ax.set_title('Boundary Layer Profiles')
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0)
 
plt.tight_layout()
plt.show()

U_true = np.sqrt(u_grid**2 + v_grid**2)

valid = (~cylinder) & (~np.isnan(U_true)) & (~np.isnan(U_full))

vmin = np.min(U_true[valid])
vmax = np.max(U_true[valid])

levels = np.linspace(vmin, vmax, 20)

fig, axs = plt.subplots(1, 2, figsize=(14,4))

U_true_plot = np.ma.array(
    U_true,
    mask=cylinder
)

cf1 = axs[0].contourf(
    X,
    Y,
    U_true_plot,
    levels=levels,
    cmap="jet",
    vmin=vmin,
    vmax=vmax
)

circle1 = plt.Circle(
    (cx, cy),
    r,
    color='k',
    fill=False,
    linestyle='--'
)

circle2 = plt.Circle(
    (cx, cy),
    r + gap_thickness,
    color='k',
    fill=False,
    linestyle='--'
)

axs[0].add_patch(circle1)
axs[0].add_patch(circle2)

axs[0].set_aspect('equal')

axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[0].set_title("Ground Truth Velocity Magnitude")

plt.colorbar(cf1, ax=axs[0], label="|U|")

U_pred_plot = np.ma.array(
    U_full,
    mask=cylinder
)

cf2 = axs[1].contourf(
    X,
    Y,
    U_pred_plot,
    levels=levels,
    cmap="jet",
    vmin=vmin,
    vmax=vmax
)

circle3 = plt.Circle(
    (cx, cy),
    r,
    color='k',
    fill=False,
    linestyle='--'
)

circle4 = plt.Circle(
    (cx, cy),
    r + gap_thickness,
    color='k',
    fill=False,
    linestyle='--'
)

axs[1].add_patch(circle3)
axs[1].add_patch(circle4)

axs[1].set_aspect('equal')

axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[1].set_title("Full PINN Velocity Magnitude")

plt.colorbar(cf2, ax=axs[1], label="|U|")

plt.tight_layout()

plt.show()

U_hybrid = U_full.copy()

U_hybrid[~gap_mask] = np.nan

U_gt = np.sqrt(u_grid**2 + v_grid**2)

U_combined = U_gt.copy()

U_combined[gap_mask] = U_hybrid[gap_mask]

U_combined = np.ma.array(
    U_combined,
    mask=cylinder
)

plt.figure(figsize=(10,4))

plt.contourf(
    X,
    Y,
    U_combined,
    levels=20,
    cmap="jet"
)

circle = plt.Circle(
    (cx, cy),
    r,
    color='k',
    fill=False,
    linestyle='--'
)

circle2 = plt.Circle(
    (cx, cy),
    r + gap_thickness,
    color='k',
    fill=False,
    linestyle='--'
)

plt.gca().add_patch(circle)
plt.gca().add_patch(circle2)

plt.gca().set_aspect('equal')

plt.xlabel("x")
plt.ylabel("y")

plt.title("PINN in Gap + Ground Truth Outside")

plt.colorbar(label="|U|")

plt.show()

theta_plot = np.linspace(0, 2*np.pi, 360)

x_cf = cx + r*np.cos(theta_plot)
y_cf = cy + r*np.sin(theta_plot)

plt.figure(figsize=(6,6))

sc = plt.scatter(
    x_cf,
    y_cf,
    c=Cf_n,
    cmap='seismic',
    s=30
)

circle = plt.Circle(
    (cx, cy),
    r,
    color='k',
    fill=False
)

plt.gca().add_patch(circle)

plt.gca().set_aspect('equal')

plt.xlabel("x")
plt.ylabel("y")

plt.title("Skin Friction Coefficient on Cylinder")

plt.colorbar(sc, label="C_f")

plt.show()

U_plot = np.ma.array(
    U_full,
    mask=cylinder
)
angles_vis = np.arange(0, 360, 20)

plt.figure(figsize=(12,5))

plt.contourf(
    X,
    Y,
    U_plot,
    levels=30,
    cmap='jet'
)

# for ang in angles_vis:

#     th = np.radians(ang)

#     x0 = cx + r*np.cos(th)
#     y0 = cy + r*np.sin(th)

#     nx = np.cos(th)
#     ny = np.sin(th)

#     tx = -ny
#     ty = nx

#     eta = np.linspace(0, 25, 120)

#     xs = x0 + nx*eta
#     ys = y0 + ny*eta

#     xs_t = torch.tensor(
#         xs,
#         dtype=torch.float32,
#         device=device
#     ).unsqueeze(1)

#     ys_t = torch.tensor(
#         ys,
#         dtype=torch.float32,
#         device=device
#     ).unsqueeze(1)

#     with torch.no_grad():
#         u, v, _ = model(xs_t, ys_t).split(1, dim=1)

#     u = u.cpu().numpy().flatten()
#     v = v.cpu().numpy().flatten()

#     ut = u*tx + v*ty

#     scale = 1

#     xb = xs + scale*ut*tx
#     yb = ys + scale*ut*ty

#     plt.plot(xb, yb, 'k', linewidth=1)

# circle = plt.Circle(
#     (cx, cy),
#     r,
#     color='white',
#     fill=False,
#     linewidth=2
# )

# plt.gca().add_patch(circle)

# plt.gca().set_aspect('equal')

# plt.xlabel("x")
# plt.ylabel("y")

# plt.title("Boundary Layer Profiles on Cylinder")

# plt.colorbar(label='|U|')

# plt.show()

#Re 41667
#Change gap height
#8 mm bin with 75% overlap