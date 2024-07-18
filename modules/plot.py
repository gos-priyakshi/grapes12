import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '-1']
y_values = [2.2, 2.6, 7.8, 129, 36898, 3855974, 3855974]
y_2 = [1.03, 0.45, 0.15, 0.01, 0.00005, 0.000000001, 0.0000000003]
# apply log scale to y_values and x values
y_values = np.log2(y_values)
y_2 = np.log2(y_2)

# Plotting the curve and save it
plt.plot(x_values, y_values, label='Dirichlet Energy 1')
plt.plot(x_values, y_2, label='Dirichlet Energy 2')
plt.title('Simple Curve Plot')
plt.xlabel('layers')
plt.ylabel('Y-axis')
#plt.ylim(1e-15, 7)
#plt.yticks(np.arange(1e-15, 7, 1e-5))
#plt.xlim(0, 70)
#plt.grid(True)  # Adds a grid for better readability
plt.show()  # Displays the plot

# Save the plot
plt.savefig('simple_curve_plot_md.png' )  # Save the plot as a PNG file
