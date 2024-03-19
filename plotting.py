import matplotlib.pyplot as plt

# Your data points


# Extract x and y coordinates
x = [point[0] for point in data]
y = [point[1] for point in data]

# Plot the line
plt.plot(x, y, marker='o')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

# Show the plot
plt.grid(True)
plt.show()
