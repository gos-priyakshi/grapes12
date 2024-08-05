import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '128']
y_gcn_s = [0.701, 0.663, 0.551, 0.386, 0.303, 0.231, 0.181]
y_gcn_f = [0.421, 0.7, 0.735, 0.704, 0.714, 0.3, 0.16]
y_res_s = [0.716, 0.688, 0.711, 0.72, 0.737, 0.742, 0.749]
y_res_f = [0.694, 0.715, 0.71, 0.718, 0.717, 0.686, 0.173]
y_gcn2_s = [0.705, 0.707, 0.738, 0.728, 0.57, 0.222, ]
y_gcn2_f = [0.381, 0.371, 0.227, 0.449, 0.283, 0.607, 0.597]
y_gat_s = [0.737, 0.768, 0.75, 0.577, 0.441, 0.285, 0.181]
y_gat_f = [0.421, 0.442, 0.369, 0.409, 0.181, 0.231, 0.181]

# Plotting the first set of curves (GCN and GAT)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_values, y_gcn_s, label='GCN-grapes', linewidth=2)
plt.plot(x_values, y_gcn_f, label='GCN', linewidth=2)
plt.plot(x_values, y_gat_s, label='GAT-grapes', linewidth=2)
plt.plot(x_values, y_gat_f, label='GAT', linewidth=2)
plt.title('Cora - GCN and GAT')
plt.xlabel('Layers')
plt.ylabel('f1_score')
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
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability

# Show the plot
plt.tight_layout()
plt.show()  # Displays the plot

# Save the plot
plt.savefig('plot2.png')  # Save the plot as a PNG file
