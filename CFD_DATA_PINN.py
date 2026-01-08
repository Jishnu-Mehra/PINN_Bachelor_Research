import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Load File
df = pd.read_csv("Flow.csv")

x = torch.tensor(df["x"].values, dtype=torch.float32, device=device).reshape(-1,1)
y = torch.tensor(df["y"].values, dtype=torch.float32, device=device).reshape(-1,1)
u = torch.tensor(df["u"].values, dtype=torch.float32, device=device).reshape(-1,1)
v = torch.tensor(df["v"].values, dtype=torch.float32, device=device).reshape(-1,1)
p = torch.tensor(df["p"].values, dtype=torch.float32, device=device).reshape(-1,1)

x.requires_grad_(True)
y.requires_grad_(True)

#Neural Network
class NavierStokesPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,3)
        )
    
    def forward(self, x, y):
        return self.net(torch.cat([x,y], dim = 1))

#NS collocation
N_NS = 10000
x_NS = torch.rand(N_NS, 1, device=device, requires_grad=True)
y_NS = torch.rand(N_NS, 1, device=device, requires_grad=True)

#Autodiff
def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs = torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

nu = 0.01 # Viscosity
u_inf = 1 # Free stream

model = NavierStokesPINN().to(device)

def data_loss():
    u_pred, v_pred, p_pred = model(x, y).split(1, dim=1)
    return (torch.mean((u_pred - u)**2) + torch.mean((v_pred - v)**2) + torch.mean((p_pred - p)**2))


def NS_loss():
    u, v, p = model(x_NS, y_NS).split(1, dim=1)

    ux = grad(u, x_NS)
    uy = grad(u, y_NS)
    vx = grad(v, x_NS)
    vy = grad(v, y_NS)
    px = grad(p, x_NS)
    py = grad(p, y_NS)

    uxx = grad(ux, x_NS)
    uyy = grad(uy, y_NS)
    vxx = grad(vx, x_NS)
    vyy = grad(vy, y_NS)

    f_cont = ux + vy
    f_mom_x = u*ux + v*uy + px - nu*(uxx + uyy)
    f_mom_y = u*vx + v*vy + py - nu*(vxx + vyy)

    return (torch.mean(f_cont**2) + torch.mean(f_mom_x**2) + torch.mean(f_mom_y**2))

def loss_calc(lambda_data, lambda_NS):
    return ((lambda_data * data_loss()) + (lambda_NS * NS_loss()))

#Hyperparameter
lambda_data, lambda_NS = 1, 0.1

#Adam Optimizer
adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Adam")
for epoch in range(2000):
    adam_optimizer.zero_grad()
    loss = loss_calc(lambda_data, lambda_NS)
    loss.backward()
    adam_optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

#L-BFGS optimizer
lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, history_size=50, line_search_fn='strong_wolfe')

def closure():
    lbfgs_optimizer.zero_grad()
    loss = loss_calc(lambda_data, lambda_NS)
    loss.backward()
    return loss

print("LBFGS")
for epoch in range(8):
    loss = lbfgs_optimizer.step(closure)
    print(f"Epoch: {epoch}, Loss: {loss}")

# Plot
nx, ny = 100, 100
x = torch.linspace(df['x'].min(), df['x'].max(), nx, device=device)
y = torch.linspace(df['y'].min(), df['y'].max(), ny, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")

u, v, _ = model(X.reshape(-1,1), Y.reshape(-1,1)).split(1, dim=1)
X, Y = X.cpu().numpy(), Y.cpu().numpy()
u = u.detach().cpu().reshape(nx,ny)
v = v.detach().cpu().reshape(nx,ny)
V = np.sqrt(u**2 + v**2)

plt.figure(figsize=(10,5))
cont = plt.contourf(X, Y, V, levels=50, cmap='rainbow')
plt.colorbar(cont, label='Velocity')
plt.quiver(X[::10,::10], Y[::10,::10], u[::10,::10], v[::10,::10], alpha=0.5)

plt.title("Fluid Flow(PINN Solution)")
plt.axis('equal')
plt.show()