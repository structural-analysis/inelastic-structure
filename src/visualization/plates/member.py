import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.settings import settings

# General settings
increment = 419  # Set the increment number here
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
def plot_members(nodes, members, yield_points, plastic_points, highlight_node=None, highlight_yield=None):
    plt.figure(figsize=(5, 5))
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

    # Plot nodes
    plt.scatter(nodes['x'], nodes['y'], color='#72858f', marker='o', s=20, label='Nodes')

    # Plot yield points
    plt.scatter(yield_points['x'], yield_points['y'], color='#4bc8fa', marker='x', s=15, label='Yield Points')

    # Highlight a specific node if provided
    if highlight_node is not None:
        node = nodes[nodes['num'] == highlight_node]
        if not node.empty:
            plt.scatter(node['x'], node['y'], color='#394247', marker='D', s=50, label='Displaced Node')
    
    # Highlight a specific yield point if provided
    if highlight_yield is not None:
        yld_point = yield_points[yield_points['num'] == highlight_yield]
        if not yld_point.empty:
            plt.scatter(yld_point['x'], yld_point['y'], color='#23bbf7', marker='P', s=70, label=f'Monitored Yield Point')

    plt.legend(loc="lower left", bbox_to_anchor=(+0.02, +0.02), framealpha=0.95)
    plt.show()

def plot(highlight_node=None, highlight_yield=None):
    nodes = load_nodes(nodes_path)
    members = pd.read_csv(members_path)
    yield_points = load_yield_points(yield_points_path)
    plastic_points = load_plastic_points(plastic_points_path)
    plot_members(nodes, members, yield_points, plastic_points, highlight_node, highlight_yield)

if __name__ == "__main__":
    highlight_node = 37  # Change this to highlight a different node
    highlight_yield = 21  # Change this to highlight a different yield point
    plot(highlight_node, highlight_yield)