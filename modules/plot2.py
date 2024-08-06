import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '128']
y_gcn_s = [0.763, 0.754, 0.711, 0.445, 0.321, 0.091, 0.091]
y_gcn_f = [0.862, 0.856, 0.842, 0.832, 0.802, 0.094, 0.103]
y_res_s = [0.747, 0.793, 0.762, 0.678, 0.735, 0.877, 0.861]
y_res_f = [0.862, 0.861, 0.849, 0.839, 0.857, 0.855, 0.85]
y_gcn2_s = [0.802, 0.799, 0.823, 0.764, 0.741, 0.845, 0.871]
y_gcn2_f = [0.734, 0.71, 0.753, 0.729, 0.792, 0.736, 0.71]
y_gat_s = [0.828, 0.811, 0.795, 0.608, 0.46, 0.39, 0.319]
y_gat_f = [0.711, 0.746, 0.717, 0.34, 0.319, 0.319, 0.319]

# Plotting the first set of curves (GCN and GAT)
y_values = [y_gcn_s, y_gcn_f, y_res_s, y_res_f, y_gcn2_s, y_gcn2_f, y_gat_s, y_gat_f]
y_min = min(min(y) for y in y_values)
y_max = max(max(y) for y in y_values)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_values, y_gcn_s, label='GCN-grapes', linewidth=2)
plt.plot(x_values, y_gcn_f, label='GCN', linewidth=2)
plt.plot(x_values, y_gat_s, label='GAT-grapes', linewidth=2)
plt.plot(x_values, y_gat_f, label='GAT', linewidth=2)
plt.title('Cora - GCN and GAT')
plt.xlabel('Layers')
plt.ylabel('f1_score')
plt.ylim(y_min, y_max) 
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability

# Plotting the second set of curves (GCN2 and ResNet)
plt.subplot(1, 2, 2)
plt.plot(x_values, y_res_s, label='ResNet-grapes', linewidth=2)
plt.plot(x_values, y_res_f, label='ResNet', linewidth=2)
plt.plot(x_values, y_gcn2_s, label='GCN2-grapes', linewidth=2)
plt.plot(x_values, y_gcn2_f, label='GCN2', linewidth=2)
plt.title('Cora - ResNet and GCN2')
plt.xlabel('Layers')
plt.ylabel('f1_score')
plt.ylim(y_min, y_max) 
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability

# Show the plot
plt.tight_layout()
plt.show()  # Displays the plot

# Save the plot
plt.savefig('plot2.png')  # Save the plot as a PNG file
