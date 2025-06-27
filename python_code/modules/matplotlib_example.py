
import matplotlib.pyplot as plt
import numpy as np

# Simple Line Plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 2]
plt.plot(x, y)
plt.xlabel(\'X-axis\')
plt.ylabel(\'Y-axis\')
plt.title(\'Simple Line Plot\')
plt.grid(True)
plt.savefig(\'simple_line_plot.png\') # Save plot instead of showing

# Scatter Plot
plt.figure() # Create a new figure for the next plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
plt.scatter(x_scatter, y_scatter, color=\'red\', marker=\'o\')
plt.xlabel(\'Random X\')
plt.ylabel(\'Random Y\')
plt.title(\'Scatter Plot\')
plt.savefig(\'scatter_plot.png\') # Save plot instead of showing

# Bar Chart
plt.figure()
categories = [\'A\', \'B\', \'C\', \'D\']
values = [20, 35, 30, 25]
plt.bar(categories, values, color=\'skyblue\')
plt.xlabel(\'Categories\')
plt.ylabel(\'Values\')
plt.title(\'Bar Chart\')
plt.savefig(\'bar_chart.png\') # Save plot instead of showing

# Histogram
plt.figure()
data_hist = np.random.randn(1000) # 1000 random numbers from a normal distribution
plt.hist(data_hist, bins=30, color=\'lightgreen\', edgecolor=\'black\')
plt.xlabel(\'Value\')
plt.ylabel(\'Frequency\')
plt.title(\'Histogram\')
plt.savefig(\'histogram.png\') # Save plot instead of showing

# Subplots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(x, y, color=\'blue\')
plt.title(\'Plot 1\')

plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.scatter(x_scatter, y_scatter, color=\'purple\')
plt.title(\'Plot 2\')

plt.tight_layout() # Adjust layout to prevent overlapping
plt.savefig(\'subplots.png\') # Save plot instead of showing


