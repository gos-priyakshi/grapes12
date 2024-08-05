import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '128']
y_gcn_s = [0.86, 0.845, 0.828, 0.836, 0.869, 0.413, 0.413]
y_gcn_f = [0.866, 0.838, 0.84, 0.836, 0.709, 0.407, 0.407]
y_res_s = [0.857, 0.834, 0.85, 0.844, 0.88, 0.873, 0.877]
y_res_f = [0.856, 0.847, 0.838, 0.843, 0.851, 0.859, 0.18]
y_gcn2_s = [0.876, 0.871, 0.881, 0.873, 0.888, 0.884, 0.407]
y_gcn2_f = [0.85, 0.847, 0.833, 0.828, 0.789, 0.777, 0.413]
y_gat_s = [0.87, 0.856, 0.841, 0.839, 0.838, 0.407, 0.407]
y_gat_f = [0.847, 0.848, 0.832, 0.838, 0.796, 0.407, 0.407]

# Plotting the first set of curves (GCN and GAT)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_values, y_gcn_s, label='GCN-grapes', linewidth=2)
plt.plot(x_values, y_gcn_f, label='GCN', linewidth=2)
plt.plot(x_values, y_gat_s, label='GAT-grapes', linewidth=2)
plt.plot(x_values, y_gat_f, label='GAT', linewidth=2)
plt.title('Pubmed - GCN and GAT')
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
plt.title('Pubmed - ResNet and GCN2')
plt.xlabel('Layers')
plt.ylabel('f1_score')
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability

# Show the plot
plt.tight_layout()
plt.show()  # Displays the plot

# Save the plot
plt.savefig('plot2.png')  # Save the plot as a PNG file
