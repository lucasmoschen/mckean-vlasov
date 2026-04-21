import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import scienceplots

from main import McKeanVlasovSolver

plt.style.use('science')

# Define the x-axis for the reconstruction (uniform in [0, 2pi])
x = np.linspace(0, 2*np.pi, 500)

def mu_0_mixed(x, a1=4.0, b1=2.0, a2=2.0, b2=10.0):
    alpha_param1, beta_param1 = a1, b1
    Z1 = (2 * np.pi)**(alpha_param1 + beta_param1 - 1) * beta(alpha_param1, beta_param1)
    alpha_param2, beta_param2 = a2, b2
    Z2 = (2 * np.pi)**(alpha_param2 + beta_param2 - 1) * beta(alpha_param2, beta_param2)

    result = 0.5*(x**(alpha_param1 - 1) * (2 * np.pi - x)**(beta_param1 - 1)) / Z1
    result += 0.5*(x**(alpha_param2 - 1) * (2 * np.pi - x)**(beta_param2 - 1)) / Z2
    
    return result

# Define two custom color maps: from grey to red (uncontrolled) and grey to blue (controlled)
cmap_uncontrolled = LinearSegmentedColormap.from_list("grey_red", ["grey", "red"])
cmap_controlled   = LinearSegmentedColormap.from_list("grey_blue", ["grey", "blue"])

V = lambda x: np.zeros_like(x)
W2 = lambda x: -5*np.cos(x)

alpha1 = lambda x: np.sin(x)
alpha2 = lambda x: np.cos(x)
alpha3 = lambda x: np.sin(2*x)
alpha4 = lambda x: np.cos(2*x)

nabla_alpha1 = lambda x: np.cos(x)
nabla_alpha2 = lambda x: -np.sin(x)
nabla_alpha3 = lambda x: 2 * np.cos(2*x)
nabla_alpha4 = lambda x: -2 * np.sin(2*x)

solver2 = McKeanVlasovSolver(L=50, d=2*np.pi, V=V, alpha=[alpha1, alpha2, alpha3, alpha4], 
                            W=W2, mu_0=mu_0_mixed, min_fourier_samples=2000, delta=-0.0001, 
                            grad_alpha=[nabla_alpha1, nabla_alpha2, nabla_alpha3, nabla_alpha4], 
                            state_weight=1000, sigma=0.5)

t_max = 1
num_snapshots = 20
t_eval = np.linspace(0, t_max, num_snapshots)
solution2_c = solver2.solve_control_problem(t_span=(0, t_max), t_eval=t_eval)
solution2_u = solver2.nonlinear_uncontrolled_solver_y(t_span=(0, t_max), t_eval=t_eval)

# Select indices uniformly from the solution time array
indices_u = np.linspace(0, len(solution2_u.t) - 1, num_snapshots, dtype=int)
indices_c = np.linspace(0, len(solution2_c.t) - 1, num_snapshots, dtype=int)

# Custom colormaps
cmap_uncontrolled = LinearSegmentedColormap.from_list("grey_red", ["grey", "red"])
cmap_controlled   = LinearSegmentedColormap.from_list("grey_blue", ["grey", "blue"])

# Plot parameters
num_snapshots = len(solution2_u.t)
lw_primary = 6   # for initial & final
lw_secondary = 3 # for intermediates
alpha_secondary = 0.7

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(16, 3), constrained_layout=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.6)

# Panel 1: Initial Model
axes[0].plot(x, solver2.reconstruction(solution2_u.y[:, 0] + solver2.bar_mu_k, x),
             color='black', linewidth=lw_primary)
axes[0].axis('off')

# Panel 2: Natural Evolution
for i in range(num_snapshots):
    y = solver2.reconstruction(solution2_u.y[:, i] + solver2.bar_mu_k, x)
    if i == 0:
        axes[1].plot(x, y, color='black', linewidth=lw_primary, zorder=2, alpha=0.5)
    elif i == num_snapshots - 1:
        axes[1].plot(x, y, color='red', linewidth=lw_primary+2, zorder=2)
    else:
        frac = i / (num_snapshots - 1)
        axes[1].plot(x, y, color=cmap_uncontrolled(frac),
                     linewidth=lw_secondary, alpha=alpha_secondary, zorder=1)
axes[1].axis('off')

# Panel 3: Guided Evolution
for i in range(num_snapshots):
    y = solver2.reconstruction(solution2_c.y[:, i] + solver2.bar_mu_k, x)
    if i == 0:
        axes[2].plot(x, y, color='black', linewidth=lw_primary, zorder=2, alpha=0.5)
    elif i == num_snapshots - 1:
        axes[2].plot(x, y, color='blue', linewidth=lw_primary+2, zorder=2)
    else:
        frac = i / (num_snapshots - 1)
        axes[2].plot(x, y, color=cmap_controlled(frac),
                     linewidth=lw_secondary, alpha=alpha_secondary, zorder=1)
axes[2].axis('off')

plt.show()
