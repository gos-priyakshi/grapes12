import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = ['2', '4', '8', '16', '32', '64']
y_values = [5.4, 10, 7.3, 3.40, 0.37, 5.4e-12]

# apply log scale to y_values and x values
y_values = np.log10(y_values)

# Plotting the curve and save it
plt.plot(x_values, y_values, marker='o')  # 'marker' is optional, it adds circles at each point
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
