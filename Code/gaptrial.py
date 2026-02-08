import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def u_analytical(y, noise_level = 0.1):
    val = 1 - y**2
    if noise_level > 0:
        # Generate Gaussian noise with mean 0 and standard deviation 'noise_level'
        noise = torch.randn_like(val) * noise_level
        return val + noise
    return val

x_min, x_max = 0.4, 0.6
y_min, y_max = -1.0, 0.0

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

#Autodiff
def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs = torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

nu = 0.01 # Viscosity
model = NavierStokesPINN().to(device)

def NS_loss():
    #NS collocation inside the gap
    N_NS = 5000
    x_NS = (x_min + (x_max - x_min) * torch.rand(N_NS,1,device=device)).requires_grad_(True)
    y_NS = (y_min + (y_max - y_min) * torch.rand(N_NS,1,device=device)).requires_grad_(True)
    
    u, v, p = model(x_NS, y_NS).split(1, dim=1)

    ux = grad(u, x_NS); uy = grad(u, y_NS)
    vx = grad(v, x_NS); vy = grad(v, y_NS)
    px = grad(p, x_NS); py = grad(p, y_NS)

    uxx = grad(ux, x_NS); uyy = grad(uy, y_NS)
    vxx = grad(vx, x_NS); vyy = grad(vy, y_NS)

    f_cont = ux + vy
    f_mom_x = u*ux + v*uy + px - nu*(uxx + uyy)
    f_mom_y = u*vx + v*vy + py - nu*(vxx + vyy)

    return (torch.mean(f_cont**2) + torch.mean(f_mom_x**2) + torch.mean(f_mom_y**2))

def BC_loss():
    #BC collocation on 4 sides of the gap
    N_BC = 500
    
    # x_left = torch.ones(N_BC,1,device=device) * x_min
    # y_left = (y_min + (y_max - y_min) * torch.rand(N_BC,1,device=device))
    
    # x_right = torch.ones(N_BC,1,device=device) * x_max
    # y_right = (y_min + (y_max - y_min) * torch.rand(N_BC,1,device=device))
    
    x_top = (x_min + (x_max - x_min) * torch.rand(N_BC,1,device=device))
    y_top = torch.ones(N_BC,1,device=device) * y_max
    
    x_bot = (x_min + (x_max - x_min) * torch.rand(N_BC,1,device=device))
    y_bot = torch.ones(N_BC,1,device=device) * y_min

    # u_l, v_l, _ = model(x_left, y_left).split(1, dim=1)
    # u_r, v_r, _ = model(x_right, y_right).split(1, dim=1)
    u_t, v_t, _ = model(x_top, y_top).split(1, dim=1)
    u_b, v_b, _ = model(x_bot, y_bot).split(1, dim=1)

    # target_u_l = u_analytical(y_left)
    # target_u_r = u_analytical(y_right)
    target_u_t = u_analytical(y_top)
    target_u_b = u_analytical(y_bot)

    # loss_u = torch.mean((u_l - target_u_l)**2) + torch.mean((u_r - target_u_r)**2) + \
    #          torch.mean((u_t - target_u_t)**2) + torch.mean((u_b - target_u_b)**2)
    loss_u = torch.mean((u_t - target_u_t)**2) + torch.mean((u_b - target_u_b)**2)
             
    # loss_v = torch.mean(v_l**2) + torch.mean(v_r**2) + torch.mean(v_t**2) + torch.mean(v_b**2)
    loss_v = torch.mean(v_t**2) + torch.mean(v_b**2)
             
    return loss_u + loss_v

def Data_loss():
    N_data = 10
    x_d = (x_min + (x_max - x_min) * torch.rand(N_data, 1, device=device))
    y_d = (y_min + (y_max - y_min) * torch.rand(N_data, 1, device=device))
    
    u_pred, v_pred, _ = model(x_d, y_d).split(1, dim=1)
    
    target_u = u_analytical(y_d)
    return torch.mean((u_pred - target_u)**2) + torch.mean(v_pred**2)

# def Data_loss():
#     x_wrong = torch.tensor([[ (x_min + x_max)/2 ]], device=device)
#     y_wrong = torch.tensor([[ (y_min + y_max)/2 ]], device=device)
    
#     u_pred, v_pred, _ = model(x_wrong, y_wrong).split(1, dim=1)
    
#     # We are telling the AI that the velocity is 5.0 (unphysical)
#     target_u = torch.tensor([[5.0]], device=device) 
    
#     return torch.mean((u_pred - target_u)**2) + torch.mean(v_pred**2)

def loss_calc():
    return 10 * BC_loss() + NS_loss()

# Gemini code for plotting at multiple epochs
# Setup Plotting Grid
nx, ny = 200, 200
x_p = torch.linspace(0, 1, nx)
y_p = torch.linspace(-1, 1, ny)
X, Y = torch.meshgrid(x_p, y_p, indexing='ij')
mask = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)
U_background = u_analytical(Y).numpy()

# Training with Snapshots
checkpoints = [1, 200, 500, 1000]
snapshots = {}
adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1001):
    adam_optimizer.zero_grad()
    current_loss = loss_calc()
    current_loss.backward()
    adam_optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {current_loss.item()}")

    if epoch in checkpoints:
        model.eval()
        with torch.no_grad():
            u_pred, _, _ = model(
                X.reshape(-1,1).to(device),
                Y.reshape(-1,1).to(device)
            ).split(1, dim=1)

            u_pinn = u_pred.cpu().reshape(nx, ny).numpy()

            # Combined field
            U_combined = U_background.copy()
            U_combined[mask.numpy()] = u_pinn[mask.numpy()]

            # Percentage error
            epsilon = 1e-6
            error = 100 * np.abs(u_pinn - U_background) / (np.abs(U_background) + epsilon)
            error_gap = np.ma.masked_where(~mask.numpy(), error)

            snapshots[epoch] = {
                "U": U_combined,
                "error": error_gap,
                "loss": current_loss.item()
            }

        model.train()

fig, axs = plt.subplots(1, len(checkpoints), figsize=(5*len(checkpoints), 4))

for i, epoch in enumerate(checkpoints):
    data = snapshots[epoch]

    cf = axs[i].contourf(
        X.numpy(), Y.numpy(), data["U"],
        levels=10, cmap='jet'
    )

    axs[i].plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        'w--', linewidth=2
    )

    axs[i].set_title(f"Epoch {epoch}\nLoss: {data['loss']:.2e}")
    axs[i].set_xlabel("x")
    if i == 0:
        axs[i].set_ylabel("y")
    axs[i].axis('equal')
    plt.colorbar(cf, ax=axs[i])

plt.suptitle("PINN Gap Reconstruction - Velocity Field", fontsize=14)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(checkpoints), figsize=(5*len(checkpoints), 4))

for i, epoch in enumerate(checkpoints):
    data = snapshots[epoch]

    cf = axs[i].contourf(
        X.numpy(), Y.numpy(), data["error"],
        levels=20, cmap='Blues'
    )

    axs[i].plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        'k--', linewidth=2
    )

    axs[i].set_title(f"Epoch {epoch} - Error (%)")
    axs[i].set_xlabel("x")
    if i == 0:
        axs[i].set_ylabel("y")
    axs[i].axis('equal')
    plt.colorbar(cf, ax=axs[i])

plt.suptitle("PINN vs Analytical Solution - Percentage Error", fontsize=14)
plt.tight_layout()
plt.show()

# #Adam Optimizer
# adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print("Adam")
# for epoch in range(501):
#     adam_optimizer.zero_grad()
#     loss = loss_calc()
#     loss.backward()
#     adam_optimizer.step()
#     if epoch % 200 == 0:
#         print(f"Epoch: {epoch}, Loss: {loss.item()}")

# # Plot
# nx, ny = 200, 200
# x = torch.linspace(0,1,nx)
# y = torch.linspace(-1,1,ny)
# X, Y = torch.meshgrid(x, y, indexing='ij')

# mask = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)

# U_background = u_analytical(Y).numpy()

# model.eval()
# with torch.no_grad():
#     X_full = X.reshape(-1,1).to(device)
#     Y_full = Y.reshape(-1,1).to(device)

#     u_pred, _, _ = model(X_full, Y_full).split(1, dim=1)
#     U_pinn = u_pred.cpu().reshape(nx, ny).numpy()

# U_combined = U_background.copy()
# U_combined[mask] = U_pinn[mask]

# plt.figure(figsize=(10,6))
# cf = plt.contourf(X.numpy(), Y.numpy(), U_combined, levels=10, cmap='jet')
# plt.colorbar(cf, label="Velocity u")

# #Draw box
# plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'w-', linewidth=2)

# plt.title("Full Analytical Field with PINN Gap Solution")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('equal')
# plt.show()