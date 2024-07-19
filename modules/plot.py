import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64', '128', '-1']
y_values = [1.136, 0.53, 0.152, 0.0068, 0.000022, 3.057e-10, 4.57e-20, 5.69e-21]
y_2 = [1.149, 0.6264, 0.154, 0.009, 0.000039, 6.69e-10, 2e-19, 5.48e-20]
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