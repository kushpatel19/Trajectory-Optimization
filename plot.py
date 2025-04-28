import matplotlib.pyplot as plt
import numpy as np

# ====== Load your real cost arrays here ======

# Example (replace by actual costs collected)
mppi_costs = np.load('mppi_costs.npy')  # Or directly use the array if you saved
ilqr_costs = np.load('ilqr_costs.npy')

iterations_mppi = np.arange(len(mppi_costs))
iterations_ilqr = np.arange(len(ilqr_costs))

# ====== Plotting the Cost over Iterations ======
plt.figure(figsize=(8,5))
plt.plot(iterations_mppi, mppi_costs, label='MPPI Controller', linewidth=2)
plt.plot(iterations_ilqr, ilqr_costs, label='iLQR Controller', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Trajectory Cost')
plt.title('Cost over Iterations: MPPI vs iLQR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('cost_plot.png', dpi=300)
plt.show()
