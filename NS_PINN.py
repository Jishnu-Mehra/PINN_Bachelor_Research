import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
x_NS = torch.rand(N_NS, 1, device=device, requires_grad=True)
y_NS = torch.rand(N_NS, 1, device=device, requires_grad=True)

#Boundary collocation
N_b = 2000
x_b = torch.rand(N_b, 1, device=device, requires_grad=True)

#y belongs to [0,1]
y_bottom = torch.zeros(N_b, 1, device=device, requires_grad=True)
y_top = torch.ones(N_b, 1, device=device, requires_grad=True) 

#Autodiff
def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs = torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

#NS residuals
nu = 0.01 # Viscosity

model = NavierStokesPINN().to(device)

#Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    optimizer.zero_grad()
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

    #BC no slip
    u_bottom, v_bottom, _ = model(x_b, y_bottom).split(1, dim=1)
    u_top, v_top, _ = model(x_b, y_top).split(1, dim=1)

    #BC Loss
    loss_bc = (torch.mean(u_bottom**2) + torch.mean(v_bottom**2) + torch.mean(u_top**2) + torch.mean(v_top**2))

    #NS Loss
    loss_NS = (torch.mean(f_cont**2) + torch.mean(f_mom_x**2) + torch.mean(f_mom_y**2))

    #Total loss
    loss = loss_bc + loss_NS
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epochs: {epoch}, Loss: {loss}")

# Forward pass for plotting
u, v, p = model(x_NS, y_NS).split(1, dim=1)

# Detach and move to CPU
u = u.detach().cpu()
v = v.detach().cpu()
y = y_NS.detach().cpu()

plt.scatter(y, u, s=5)
plt.xlabel("y")
plt.ylabel("u")
plt.title("Velocity profile u(y)")
plt.show()