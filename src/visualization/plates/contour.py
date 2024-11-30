import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path

# General settings
example_name = "plate-curved-fine"
selected = "von_mises_moment"  # nodal_disp, von_mises_moment
increment = 194  # 82 kn/m2
contour_levels = 9
min_limit = 50000  # Set your minimum limit here
max_limit = 150000  # Set your maximum limit here
colormap_choice = "Spectral"  # "Spectral" or "Grays" for white-to-black
inputs_dir = "input/examples/"
outputs_dir = "output/examples/"

# Paths for input files
nodes_path = os.path.join(inputs_dir, example_name, 'nodes.csv')
displacements_path = os.path.join(outputs_dir, example_name, str(increment), 'nodal_disp/0.csv')
elements_path = os.path.join(inputs_dir, example_name, 'members/plates.csv')
moments_path = os.path.join(outputs_dir, example_name, str(increment), 'nodal_moments/0.csv')

# Function to load node coordinates
def load_nodes(nodes_path):
    nodes = pd.read_csv(nodes_path)
    x = nodes['x'].values
    y = nodes['y'].values
    return x, y

# Function to load displacement data
def load_displacement(displacements_path):
    displacements = pd.read_csv(displacements_path, header=None)
    dz = displacements.iloc[::3, 0].values  # Every third element (dz)
    return dz

# Function to load moment data and calculate von Mises moments
def load_von_mises_moment(moments_path):
    moments = pd.read_csv(moments_path, header=None)
    mx = moments.iloc[::3, 0].values  # Mx
    my = moments.iloc[1::3, 0].values  # My
    mxy = moments.iloc[2::3, 0].values  # Mxy
    
    # Calculate the von Mises moment using the formula: sqrt(Mx^2 + My^2 - Mx*My + 3*Mxy^2)
    von_mises = np.sqrt(mx**2 + my**2 - mx * my + 3 * mxy**2)
    return von_mises

# Function to create the interpolation grid
def create_grid(x, y, values, mask_elements=True, elements=None, grid_size=200):
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), values, (xi, yi), method='cubic')
    
    # Apply masking if necessary
    if mask_elements and elements is not None:
        mask = np.full(xi.shape, False)
        for _, row in elements.iterrows():
            node_indices = list(map(int, row['nodes'].split('-')))
            element_x = x[node_indices]
            element_y = y[node_indices]
            polygon = Path(np.column_stack((element_x, element_y)))
            points = np.column_stack((xi.ravel(), yi.ravel()))
            inside = polygon.contains_points(points).reshape(xi.shape)
            mask = np.logical_or(mask, inside)
        zi[~mask] = np.nan
    
    # Apply limits to the zi values only if they are specified
    if min_limit is not None or max_limit is not None:
        zi = np.clip(zi, min_limit if min_limit is not None else zi.min(), 
                     max_limit if max_limit is not None else zi.max())
    
    return xi, yi, zi

# Function to plot the contour of a given response
def plot_contour(xi, yi, zi, response_name, response_title, x, y, elements, contour_levels):
    # Determine the minimum and maximum limits for the contour levels
    organic_min = np.nanmin(zi)
    organic_max = np.nanmax(zi)

    # Handle cases where the entire zi array is NaN
    if np.isnan(organic_min) or np.isnan(organic_max):
        raise ValueError("Cannot plot contour: all zi values are NaN. Check the input data and grid settings.")

    # Use specified limits if provided, otherwise use organic values
    actual_min = min_limit if min_limit is not None else organic_min
    actual_max = max_limit if max_limit is not None else organic_max

    # Create levels based on the actual limits
    levels = np.linspace(actual_min, actual_max, contour_levels)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, zi, levels=levels, cmap=colormap_choice)

    # Create the colorbar with a title and smaller size
    cbar = plt.colorbar(contour, ticks=levels, shrink=0.4, aspect=4, pad=0.05)
    cbar.set_label(f'{response_name}')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f'{response_name} {response_title}', fontsize=10, labelpad=10)
    plt.title(f'Contour of {response_name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')

    # Add padding (5% of the range)
    padding_x = 0.05 * (x.max() - x.min())
    padding_y = 0.05 * (y.max() - y.min())
    plt.xlim(x.min() - padding_x, x.max() + padding_x)
    plt.ylim(y.min() - padding_y, y.max() + padding_y)

    # Draw elements on the plot
    for _, row in elements.iterrows():
        node_indices = list(map(int, row['nodes'].split('-')))
        element_x = x[node_indices]
        element_y = y[node_indices]
        plt.plot(np.append(element_x, element_x[0]), np.append(element_y, element_y[0]), 'k-', linewidth=0.5)
    
    plt.show()

# Main function to load data and create the plot for a given response
def plot_response(response_type='nodal_disp'):
    # Load node coordinates and element data
    x, y = load_nodes(nodes_path)
    elements = pd.read_csv(elements_path)
    
    # Determine which response to plot
    if response_type == 'nodal_disp':
        values = load_displacement(displacements_path)
        response_name = 'Vertical Displacement (dz)'
        response_title = '(U3, m)'
    elif response_type == 'von_mises_moment':
        values = load_von_mises_moment(moments_path)
        response_name = 'Von Mises Moment'
        response_title = '(Mises-M, Pa)'
    else:
        raise ValueError(f"Unknown response type: {response_type}")

    # Create grid and interpolate the values
    xi, yi, zi = create_grid(x, y, values, mask_elements=True, elements=elements)
    
    # Plot the contour
    plot_contour(xi, yi, zi, response_name, response_title, x, y, elements, contour_levels)

if __name__ == "__main__":
    plot_response(selected)
