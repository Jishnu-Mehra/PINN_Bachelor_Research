import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


x_axis = np.load("Data/x.npy")
y_axis = np.load("Data/y.npy")
u_data = np.load("Data/u.npy")
v_data = np.load("Data/v.npy")

# assert u_data.shape == (len(x_axis), len(y_axis))
# assert v_data.shape == (len(x_axis), len(y_axis))

X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")

# plt.figure(figsize=(8, 2))
# plt.contourf(X, Y, u_data, levels=20, cmap="jet")
# plt.xlabel("x (streamwise)")
# plt.ylabel("y (wall-normal)")
# plt.title("Ground truth Blasius boundary layer")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

x_min, x_max = x_axis.min() + 1, x_axis.max() - 1
y_min = y_axis.min()
y_max = 0.1 * y_axis.max()

mask = (Y >= y_min) & (Y <= y_max) & (X >= x_min) & (X <= x_max)

x_t = torch.tensor(x_axis, device=device)
y_t = torch.tensor(y_axis, device=device)

def sample_background(x, y):
    ix = torch.bucketize(x.squeeze(), x_t) - 1
    iy = torch.bucketize(y.squeeze(), y_t) - 1

    ix = torch.clamp(ix, 0, len(x_axis)-1)
    iy = torch.clamp(iy, 0, len(y_axis)-1)

    u = torch.tensor(u_data[ix.cpu(), iy.cpu()], device=device).unsqueeze(1)
    v = torch.tensor(v_data[ix.cpu(), iy.cpu()], device=device).unsqueeze(1)
    return u, v

class BL_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

def grad(f, x):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True
    )[0]

model = BL_PINN().to(device)

nu = 1e-5

def NS_loss():
    N = 2000
    x = (x_min + (x_max-x_min)*torch.rand(N,1,device=device)).requires_grad_(True)
    y = (y_min + (y_max-y_min)*torch.rand(N,1,device=device)).requires_grad_(True)

    u, v, p = model(x,y).split(1,dim=1)

    ux = grad(u,x)
    uy = grad(u,y)
    vy = grad(v,y)
    uyy = grad(uy,y)
    px = grad(p, x)
    py = grad(p, y)

    cont = ux + vy
    momx  = u*ux + v*uy - nu*uyy + (1/1.225)*px
    momy = (1/1.225)*py

    return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)

def wall_bc():
    x = (x_min + (x_max-x_min)*torch.rand(500,1,device=device))
    y = torch.zeros_like(x)

    u,v,_ = model(x,y).split(1,dim=1)
    return torch.mean(u**2) + torch.mean(v**2)

def top_bc():
    x = (x_min + (x_max - x_min) * torch.rand(500, 1, device=device))
    y = torch.ones_like(x) * y_max

    u_p, v_p,_ = model(x, y).split(1, dim=1)
    u_inf, v_inf = sample_background(x, y)

    return torch.mean((u_p - u_inf)**2) + torch.mean((v_p - v_inf)**2)

def loss():
    return NS_loss() + 10 * wall_bc() + 10 * top_bc()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoints = [1, 200, 500, 1000]
snapshots = {}

for epoch in range(1001):
    optimizer.zero_grad()
    L = loss()
    L.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {L.item():.3e}")

    if epoch in checkpoints:
        with torch.no_grad():
            Xg = torch.tensor(X[mask], dtype=torch.float32, device=device).reshape(-1,1)
            Yg = torch.tensor(Y[mask], dtype=torch.float32, device=device).reshape(-1,1)
            u_pred,_,_ = model(Xg,Yg).split(1,dim=1)

            U_comb = u_data.copy()
            U_comb[mask] = u_pred.cpu().numpy().flatten()

            err = 100*np.abs(U_comb-u_data)/(np.abs(u_data)+1e-6)
            snapshots[epoch] = (U_comb, err)


fig, axs = plt.subplots(1, len(checkpoints), figsize=(6*len(checkpoints), 3))

for i, e in enumerate(checkpoints):
    cf = axs[i].contourf(
        X, Y, snapshots[e][0],
        levels=20, cmap="jet"
    )
    axs[i].set_title(f"Epoch {e}")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")
    plt.colorbar(cf, ax=axs[i])

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(checkpoints), figsize=(6*len(checkpoints), 3))

for i, e in enumerate(checkpoints):
    cf = axs[i].contourf(
        X, Y, snapshots[e][1],
        levels=20, cmap="Blues"
    )
    axs[i].set_title(f"Error % (Epoch {e})")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")
    plt.colorbar(cf, ax=axs[i])

plt.tight_layout()
plt.show()