import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from main import McKeanVlasovSolver

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

G = lambda x: np.zeros_like(x)
W2 = lambda x: -5*np.cos(x)

alpha1 = lambda x: np.sin(x)
alpha2 = lambda x: np.cos(x)
alpha3 = lambda x: np.sin(2*x)
alpha4 = lambda x: np.cos(2*x)

nabla_alpha1 = lambda x: np.cos(x)
nabla_alpha2 = lambda x: -np.sin(x)
nabla_alpha3 = lambda x: 2 * np.cos(2*x)
nabla_alpha4 = lambda x: -2 * np.sin(2*x)

solver2 = McKeanVlasovSolver(L=50, d=2*np.pi, G=G, alpha=[alpha1, alpha2, alpha3, alpha4], 
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

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16/2, 9/2), dpi=300)

# Plot for Uncontrolled Evolution
for i in range(num_snapshots):
    distribution = solver2.reconstruction(solution2_u.y[:, i]+solver2.bar_mu_k, x)
    if i == 0:
        # Initial condition: black, full opacity
        ax1.plot(x, distribution, color="black", alpha=1.0, linewidth=5, label="Initial Distribution")
    elif i == num_snapshots - 1:
        # Final condition: red, full opacity
        color = cmap_uncontrolled(1.0)
        ax1.plot(x, distribution, color=color, alpha=1.0, linewidth=5, label="Final Distribution")
    else:
        frac = i / (num_snapshots - 1)
        color = cmap_uncontrolled(frac)
        ax1.plot(x, distribution, color=color, alpha=0.5, linewidth=5)

ax1.set_title('Uncontrolled Evolution', fontsize=18)
ax1.set_xlabel('x', fontsize=15)
ax1.grid(True, linestyle='--', linewidth=0.75, color='grey', alpha=0.7)
ax1.legend(fontsize=14)

# Plot for Controlled Evolution
for i in range(num_snapshots):
    distribution = solver2.reconstruction(solution2_c.y[:, i]+solver2.bar_mu_k, x)
    if i == 0:
        # Initial condition: black, full opacity
        ax2.plot(x, distribution, color="black", alpha=1.0, linewidth=5, label="Initial Distribution")
    elif i == num_snapshots - 1:
        # Final condition: blue, full opacity
        color = cmap_controlled(1.0)
        ax2.plot(x, distribution, color=color, alpha=1.0, linewidth=5, label="Final Distribution")
    else:
        frac = i / (num_snapshots - 1)
        color = cmap_controlled(frac)
        ax2.plot(x, distribution, color=color, alpha=0.5, linewidth=5)

ax2.set_title('Controlled Evolution', fontsize=18)
ax2.set_xlabel('x', fontsize=14)
ax2.grid(True, linestyle='--', linewidth=0.75, color='grey', alpha=0.7)
ax2.legend(fontsize=14)

plt.tight_layout()
plt.savefig("poster_plot_2.pdf", bbox_inches="tight")
plt.show()
