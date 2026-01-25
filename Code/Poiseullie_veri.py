import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def u_analytical(y):
    return 1 - y**2

def v_analytical(y):
    return torch.zeros_like(y)

x_min, x_max = 0.4, 0.6
y_min, y_max = -1, 0


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
u_inf, v_inf = 1, 0
model = NavierStokesPINN().to(device)

def NS_loss():
    #NS collocation
    N_NS = 10000
    x_NS = torch.rand(N_NS,1,device=device,requires_grad=True)
    y_NS = -1 + 2*torch.rand(N_NS,1,device=device,requires_grad=True)
    
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

    dpdx = - 2 * nu

    return (torch.mean(f_cont**2) + torch.mean(f_mom_x**2) + torch.mean(f_mom_y**2) + torch.mean((px - dpdx)**2))

def BC_loss():
    #BC collocation
    N_BC = 2000
    x_bc = torch.rand(N_BC,1,device=device,requires_grad=True)
    y_bc_top = torch.ones(N_BC,1,device=device,requires_grad=True)
    y_bc_bot = -torch.ones(N_BC,1,device=device,requires_grad=True)

    y_in = -1 + 2*torch.rand(N_BC,1,device=device,requires_grad=True)
    x_in = torch.zeros(N_BC,1,device=device,requires_grad=True)
    y_out = -1 + 2*torch.rand(N_BC,1,device=device,requires_grad=True)
    x_out = torch.ones(N_BC,1,device=device,requires_grad=True)

    u_top, v_top, _ = model(x_bc, y_bc_top).split(1, dim=1)
    u_bot, v_bot, _ = model(x_bc, y_bc_bot).split(1, dim=1)

    loss_wall = torch.mean(u_top**2 + v_top**2 + u_bot**2 + v_bot**2)

    u_in, v_in, _ = model(x_in, y_in).split(1, dim=1)
    _, _, p_out = model(x_out, y_out).split(1, dim=1)

    u_target = 1 - y_in**2
    loss_inlet = ((torch.mean((u_in - u_target)**2) + torch.mean(v_in**2)))
    loss_outlet = torch.mean(p_out**2)

    return loss_wall + loss_inlet + loss_outlet

def loss_calc(lambda_BC, lambda_NS):
    return ((lambda_BC * BC_loss()) + (lambda_NS * NS_loss()))

#Hyperparameter
lambda_NS = 1
lambda_BC = 1

#Adam Optimizer
adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Adam")
for epoch in range(1001):
    adam_optimizer.zero_grad()
    loss = loss_calc(lambda_BC, lambda_NS)
    loss.backward()
    adam_optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

# #L-BFGS optimizer
# lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, history_size=50, line_search_fn='strong_wolfe')

# def closure():
#     lbfgs_optimizer.zero_grad()
#     loss = loss_calc(lambda_BC, lambda_NS)
#     loss.backward()
#     return loss

# print("LBFGS")
# for epoch in range(3):
#     loss = lbfgs_optimizer.step(closure)
#     print(f"Epoch: {epoch}, Loss: {loss}")

# Plot
nx, ny = 200, 200
x = torch.linspace(0,1,nx)
y = torch.linspace(-1,1,ny)
X, Y = torch.meshgrid(x, y, indexing='ij')

mask = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)

model.eval()
with torch.no_grad():
    X_full = X.reshape(-1,1).to(device)
    Y_full = Y.reshape(-1,1).to(device)

    u_pred_full, _, _ = model(X_full, Y_full).split(1, dim=1)
    U_pinn_full = u_pred_full.cpu().reshape(nx, ny)

# Analytical field
U_true = u_analytical(Y)

# Insert PINN solution into gap
U_true_masked = U_true.clone()
U_true_masked[mask] = U_pinn_full[mask]

U_plot = U_true_masked.cpu().numpy()
mask_np = mask.cpu().numpy()

plt.figure(figsize=(14,5))

# Plot 1: Analytical + PINN Gap Reconstruction
plt.subplot(1,2,1)
# Save the contourf object to variable 'cf'
cf = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), U_plot, levels=10, cmap='jet')
# Plot the white boundary line
plt.contour(X.cpu().numpy(), Y.cpu().numpy(), mask_np.astype(int), levels=[0.5], colors='white', linewidths=2)
# Pass 'cf' to colorbar so it uses the filled contours, not the line
plt.colorbar(cf, label="Velocity u") 
plt.title("Analytical Solution with PINN Gap Reconstruction")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

# Plot 2: Pure PINN Solution
plt.subplot(1,2,2)
cf2 = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), U_pinn_full.numpy(), levels=10, cmap='jet')
plt.colorbar(cf2, label="Velocity u")
plt.title("Pure PINN Solution (No Analytical Info Used)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

plt.tight_layout()
plt.show()

# Gemini code for plotting over epochs ----------------------------
# # 1. SETUP GRID
# nx, ny = 200, 200
# x = torch.linspace(0, 1, nx)
# y = torch.linspace(-1, 1, ny)
# X, Y = torch.meshgrid(x, y, indexing='ij')
# X_full = X.reshape(-1, 1).to(device)
# Y_full = Y.reshape(-1, 1).to(device)

# # 2. TRAINING WITH SNAPSHOTS
# checkpoints = [200, 500, 1000]
# snapshots = {} # This will now store {epoch: (u_data, loss_value)}

# adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# print("Starting Training...")
# for epoch in range(1001):
#     adam_optimizer.zero_grad()
#     current_loss = loss_calc(lambda_BC, lambda_NS) # Rename to avoid confusion
#     current_loss.backward()
#     adam_optimizer.step()
    
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch}, Loss: {current_loss.item():.6f}")

#     # SAVE BOTH DATA AND LOSS HERE
#     if epoch in checkpoints:
#         model.eval()
#         with torch.no_grad():
#             u_pred, _, _ = model(X_full, Y_full).split(1, dim=1)
#             # We store a tuple: (numpy_array, scalar_loss)
#             snapshots[epoch] = (u_pred.cpu().reshape(nx, ny).numpy(), current_loss.item())
#         model.train()

# # 3. VISUALIZATION
# fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# for i, epoch in enumerate(checkpoints):
#     ax = axs[i]
    
#     # Unpack the stored data
#     u_data, saved_loss = snapshots[epoch]
    
#     cf = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), u_data, levels=20, cmap='jet')
    
#     # Now 'saved_loss' is unique to that specific epoch
#     ax.set_title(f"Epoch {epoch}\nLoss: {saved_loss:.2e}")
#     ax.set_xlabel("x")
#     if i == 0: ax.set_ylabel("y")
#     ax.axis('equal')
#     plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

# plt.suptitle("PINN Convergence: Velocity Field (u) at Specific Epochs", fontsize=16)
# plt.tight_layout()
# plt.show()

# x_min, x_max = 0.4, 0.6
# y_min, y_max = -1.0, 0.0  # Note: your problem domain is y from -1 to 1

# # 2. Get the Final Prediction at 1000 Epochs
# model.eval()
# with torch.no_grad():
#     # Ensure grid is on the same device as model
#     X_full = X.reshape(-1,1).to(device)
#     Y_full = Y.reshape(-1,1).to(device)
    
#     u_pred, _, _ = model(X_full, Y_full).split(1, dim=1)
#     U_pinn = u_pred.cpu().reshape(nx, ny) # Move to CPU for numpy operations

# # 3. Create the Mask and Hybrid Plot
# # Create mask on CPU to match U_pinn and U_true
# X_cpu, Y_cpu = X.cpu(), Y.cpu()
# mask = (X_cpu > x_min) & (X_cpu < x_max) & (Y_cpu > y_min) & (Y_cpu < y_max)

# # Generate Analytical Solution
# U_true = u_analytical(Y_cpu)

# # Create the "Gap" hybrid
# U_hybrid = U_true.clone()
# U_hybrid[mask] = U_pinn[mask]

# # 4. Plotting
# plt.figure(figsize=(10, 6))

# # Main plot
# cf = plt.contourf(X_cpu.numpy(), Y_cpu.numpy(), U_hybrid.numpy(), levels=20, cmap='jet')

# # Draw the white boundary box for the gap
# mask_np = mask.numpy().astype(float)
# plt.contour(X_cpu.numpy(), Y_cpu.numpy(), mask_np, levels=[0.5], colors='white', linewidths=2)

# plt.colorbar(cf, label="Velocity u")
# plt.title("Poiseuille Flow: Analytical Solution with PINN Gap (Epoch 1000)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('equal')
# plt.show()