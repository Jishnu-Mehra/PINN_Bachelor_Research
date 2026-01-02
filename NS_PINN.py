import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
x_NS = torch.rand(N_NS, 1, device=device)
y_NS = torch.rand(N_NS, 1, device=device)

#Boundary collocation

#Inlet, Outlet
N_in = 1000
x_in = torch.zeros(N_in, 1, device=device, requires_grad=True)
y_in = torch.rand(N_in, 1, device=device, requires_grad=True)

N_out = 1000
x_out = torch.ones(N_out, 1, device=device, requires_grad=True)
y_out = torch.rand(N_out, 1, device=device, requires_grad=True)

#Walls
N_w = 1000
x_w = torch.rand(N_w, 1, device=device, requires_grad=True)
y_top = torch.ones(N_w, 1, device=device, requires_grad=True)
y_bot = torch.zeros(N_w, 1, device=device, requires_grad=True)

#Cylinder
N_cyl = 2000
c_x, c_y, r = 0.5, 0.5, 0.1
dist = (x_NS - c_x)**2 + (y_NS - c_y)**2
mask = dist >= r**2
mask = mask.squeeze()

x_NS = x_NS[mask].requires_grad_(True)
y_NS = y_NS[mask].requires_grad_(True)

theta = 2 * torch.pi * torch.rand(N_cyl, 1, device=device)
x_cyl = c_x + r * torch.cos(theta)
y_cyl = c_y + r * torch.sin(theta)

x_cyl.requires_grad_(True)
y_cyl.requires_grad_(True)

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

#Training Loop

def loss_calc():
    #Outputs split using forward
    u, v, p = model(x_NS, y_NS).split(1, dim=1)

    #First Derivatives
    ux = grad(u, x_NS)
    uy = grad(u, y_NS)
    vx = grad(v, x_NS)
    vy = grad(v, y_NS)
    px = grad(p, x_NS)
    py = grad(p, y_NS)

    #Second Derivatives
    uxx = grad(ux, x_NS)
    uyy = grad(uy, y_NS)
    vxx = grad(vx, x_NS)
    vyy = grad(vy, y_NS)

    #Continuity Equation
    f_cont = ux + vy

    #Momentum Equation
    f_mom_x = u*ux + v*uy + px - nu*(uxx+uyy)
    f_mom_y = u*vx + v*vy + py - nu*(vxx+vyy)

    #BC inlet, outlet
    u_in, v_in, _ = model(x_in, y_in).split(1, dim=1)
    _, _, p_out = model(x_out, y_out).split(1, dim=1)

    #BC no slip
    u_bot, v_bot, _ = model(x_w, y_bot).split(1, dim=1)
    u_top, v_top, _ = model(x_w, y_top).split(1, dim=1)
    u_cyl, v_cyl, _ = model(x_cyl, y_cyl).split(1, dim=1)

    #BC Loss
    loss_wall = (torch.mean(u_bot**2) + torch.mean(v_bot**2) + torch.mean(u_top**2) + torch.mean(v_top**2))
    loss_inlet = (torch.mean((u_in - u_inf)**2) + torch.mean(v_in**2))
    loss_outlet = (torch.mean(p_out**2))
    loss_cylinder = (torch.mean(u_cyl**2) + torch.mean(v_cyl**2))
    loss_bc = loss_wall + loss_inlet + loss_outlet + 10 * loss_cylinder

    #NS Loss
    loss_NS = (torch.mean(f_cont**2) + torch.mean(f_mom_x**2) + torch.mean(f_mom_y**2))

    #Total loss
    loss = loss_bc + loss_NS
    return loss

#Adam Optimizer
adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Adam")
for epoch in range(2000):
    adam_optimizer.zero_grad()
    loss = loss_calc()
    loss.backward()
    adam_optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

#L-BFGS optimizer
lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, history_size=50, line_search_fn='strong_wolfe')

def closure():
    lbfgs_optimizer.zero_grad()
    loss = loss_calc()
    loss.backward()
    return loss

print("LBFGS")
for epoch in range(8):
    loss = lbfgs_optimizer.step(closure)
    print(f"Epoch: {epoch}, Loss: {loss}")

# Plot
nx, ny = 200, 200
x = torch.linspace(0,1,nx, device=device)
y = torch.linspace(0,1,ny, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")

u, v, _ = model(X.reshape(-1,1), Y.reshape(-1,1)).split(1, dim=1)
u = u.detach().cpu().reshape(nx,ny)
v = v.detach().cpu().reshape(nx,ny)

# Mask cylinder
dist_ = (X - c_x)**2 + (Y - c_y)**2
mask_ = dist_ < r**2
u[mask_] = np.nan
v[mask_] = np.nan
X = X.cpu()
Y = Y.cpu()

plt.figure(figsize=(10,5))
cont = plt.contourf(X, Y, u, levels=50, cmap='rainbow')
plt.colorbar(cont, label='Velocity u')
plt.quiver(X, Y, u, v)
plt.gca().add_patch(plt.Circle((c_x, c_y), r, color='white'))
plt.title("Fluid Flow around Cylinder (PINN Solution)")
plt.axis('equal')
plt.show()