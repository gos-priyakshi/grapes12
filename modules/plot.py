import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '-1']
y_values = [1.03, 0.48, 0.11, 0.006, 0.00001, 1.8e-10, 5.02e-11]
y_2 = [0.2, 0.11, 0.02, 0.004, 0.00001, 4.5e-10, 4.5e-10]
#y_3 = [3.25, 3.83, 7.6, 55, 2569, 554366, 1972222, 1972222]

# Apply log scale to y_values and y_2
y_values = np.log2(y_values)
y_2 = np.log2(y_2)

# Plotting the curve
plt.plot(x_values, y_values, label='fb')
plt.plot(x_values, y_2, label='grapes')
#plt.plot(x_values, y_3, label='res')
plt.title('Simple Curve Plot')
plt.xlabel('Layers')
plt.ylabel('DE')
plt.legend()  # Add legend
plt.grid(True)  # Adds a grid for better readability
plt.show()  # Displays the plot

# Save the plot
plt.savefig('plot2.png')  # Save the plot as a PNG file