from math import log
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '128']
y_gcn_s = [0.198, 0.118, 0.051, 0.009, 0.0000137, 5.878e-8, 3.824e-16]
y_gcn_f = [1.003, 0.5115, 0.1685, 0.0143, 0.000041, 4.152e-10, 1.066e-19]
y_res_s = [0.5597, 0.6914, 0.6656, 0.6646, 0.7175, 0.664, 2.592]
y_res_f = [3.67, 3.784, 3.798, 3.879, 3.815, 3.65, 26]
y_gcn2_s = [0.1998, 0.2819, 0.3672, 0.453, 1.16, 1.3, 11]
y_gcn2_f = [1.424, 1.347, 1.488, 1.95, 2.491, 2.847, 3.147]
y_gat_s = [0.4574, 0.1042, 0.01276, 0.0003165, 3.301e-7, 7.987e-9, 8.734e-14]
y_gat_f = [0.4758, 0.1204, 0.01322, 0.0003851, 5.509e-7, 6.710e-9, 1.02e-15]
#y_3 = [3.25, 3.83, 7.6, 55, 2569, 554366, 1972222, 1972222]

# Apply log 2 scale 
y_gcn_s = np.log2(y_gcn_s)
y_gcn_f = np.log2(y_gcn_f)
y_res_s = np.log2(y_res_s)
y_res_f = np.log2(y_res_f)
y_gcn2_s = np.log2(y_gcn2_s)
y_gcn2_f = np.log2(y_gcn2_f)
y_gat_s = np.log2(y_gat_s)
y_gat_f = np.log2(y_gat_f)

# Plotting the curve
# make the lines thicker
plt.plot(x_values, y_gcn_s, label='GCN-grapes')
plt.plot(x_values, y_gcn_f, label='GCN')
plt.plot(x_values, y_res_s, label='ResNet-grapes')
plt.plot(x_values, y_res_f, label='ResNet')
plt.plot(x_values, y_gcn2_s, label='GCN2-grapes')
plt.plot(x_values, y_gcn2_f, label='GCN2')
plt.plot(x_values, y_gat_s, label='GAT-grapes')
plt.plot(x_values, y_gat_f, label='GAT')
#plt.plot(x_values, y_3, label='res')
plt.title('Cora')
plt.xlabel('Layers')
plt.ylabel('Dirichlet Energy(log(E(x)))')
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability
plt.show()  # Displays the plot

# Save the plot
plt.savefig('plot2.png')  # Save the plot as a PNG file