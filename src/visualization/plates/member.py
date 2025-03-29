import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.settings import settings

# General settings
increment = 10  # Set the increment number here
example_name = settings.example_name
inputs_dir = "input/examples/"
outputs_dir = "output/examples/"
nodes_path = os.path.join(inputs_dir, example_name, 'nodes.csv')
members_path = os.path.join(inputs_dir, example_name, 'members/plates.csv')
yield_points_path = os.path.join(outputs_dir, example_name, 'yield_points.csv')
plastic_points_path = os.path.join(outputs_dir, example_name, str(increment), 'plastic_points', '0.csv')

# Function to load node coordinates
def load_nodes(nodes_path):
    nodes = pd.read_csv(nodes_path)
    return nodes

# Function to load yield points
def load_yield_points(yield_points_path):
    yield_points = pd.read_csv(yield_points_path, header=None, names=['num', 'x', 'y'])
    return yield_points

# Function to load plastic points
def load_plastic_points(plastic_points_path):
    if os.path.exists(plastic_points_path):
        plastic_points = pd.read_csv(plastic_points_path, header=None, names=['num'])
        return plastic_points['num'].values
    return []

# Function to plot the contour of a given response
def plot_members(nodes, members, yield_points, plastic_points):
    plt.figure(figsize=(10, 8))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

    x = nodes['x'].values
    y = nodes['y'].values
    
    # Draw elements on the plot
    for _, row in members.iterrows():
        node_indices = list(map(int, row['nodes'].split('-')))
        element_x = x[node_indices]
        element_y = y[node_indices]
        plt.plot(np.append(element_x, element_x[0]), np.append(element_y, element_y[0]), 'k-', linewidth=0.5)
        
        # Plot member number at its centroid
        centroid_x = np.mean(element_x)
        centroid_y = np.mean(element_y)
        plt.text(centroid_x, centroid_y, str(row['num']), fontsize=8, ha='center', va='center', color='red')
    
    # Plot yield points
    plt.scatter(yield_points['x'], yield_points['y'], color='blue', marker='x', s=20, label='Yield Points')
    
    # Annotate yield points with their numbers
    for _, row in yield_points.iterrows():
        plt.text(row['x'], row['y'], str(int(row['num'])), fontsize=8, ha='left', va='bottom', color='blue')
    
    # Plot nodes
    plt.scatter(nodes['x'], nodes['y'], color='green', marker='o', s=10, label='Nodes')
    
    # Annotate nodes with their numbers
    for _, row in nodes.iterrows():
        plt.text(row['x'], row['y'], str(row['num']), fontsize=6, ha='right', va='top', color='green')
    
    # Plot plastic points
    plastic_yield_points = yield_points[yield_points['num'].isin(plastic_points)]
    plt.scatter(plastic_yield_points['x'], plastic_yield_points['y'], color='magenta', marker='*', s=50, label='Plastic Points')
    
    plt.legend()
    plt.show()

def plot():
    nodes = load_nodes(nodes_path)
    members = pd.read_csv(members_path)
    yield_points = load_yield_points(yield_points_path)
    plastic_points = load_plastic_points(plastic_points_path)
    plot_members(nodes, members, yield_points, plastic_points)

if __name__ == "__main__":
    plot()
