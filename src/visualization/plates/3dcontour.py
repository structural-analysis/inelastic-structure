import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

example_name = "plate-curved-fine"
increment = 597
contour_levels = 12
inputs_dir = "input/examples/"
outputs_dir = "output/examples/"

nodes_path = os.path.join(inputs_dir, example_name, 'nodes.csv')
displacements_path = os.path.join(outputs_dir, example_name, str(increment), 'nodal_disp/0.csv')
elements_path =  os.path.join(inputs_dir, example_name, 'members/plates.csv')

nodes = pd.read_csv(nodes_path)
displacements = pd.read_csv(displacements_path, header=None)
elements = pd.read_csv(elements_path)

# Extract the node numbers and coordinates
x = nodes['x'].values
y = nodes['y'].values

# Extract displacement data
dz = displacements.iloc[::3, 0].values  # Every third element starting from index 0 (dz)

# Create a grid for contour plotting
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate dz data onto the grid
zi = griddata((x, y), dz/10, (xi, yi), method='cubic')

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface with displacement values as z-coordinates
surface = ax.plot_surface(xi, yi, zi, cmap='Spectral', edgecolor='k', alpha=0.7)

# Add a color bar to represent displacement values
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Vertical Displacement (dz)')

# Draw the elements on the 3D plot
for idx, row in elements.iterrows():
    # Split the 'nodes' column to get individual node indices, and convert to integers
    node_indices = list(map(int, row['nodes'].split('-')))
    
    # Get x and y coordinates of the nodes for the current element
    element_x = x[node_indices]
    element_y = y[node_indices]
    element_z = dz[node_indices]  # Get the displacement values (z-coordinates)

    # Close the loop to draw the last line back to the first node
    ax.plot(np.append(element_x, element_x[0]), 
            np.append(element_y, element_y[0]), 
            np.append(element_z, element_z[0]), 
            'k-', linewidth=0.5)

# Annotate node numbers in 3D
for i, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, dz)):
    ax.text(x_coord, y_coord, z_coord, str(i+1), fontsize=8, color='black')

# Set plot labels and title
ax.set_title('3D Contour of Vertical Displacement (dz) with Elements')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Vertical Displacement (dz)')

plt.tight_layout()
plt.show()