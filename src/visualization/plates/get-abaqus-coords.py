from abaqus import *
from abaqusConstants import *
import mesh
import csv
part_name = 'ring-3d'
model = mdb.models['Model-1']
part = model.parts[part_name]
elements = part.elements
nodes = part.nodes
print(f"Number of elements: {len(elements)}")
print(f"Number of nodes: {len(nodes)}")
# File paths for the output files
element_nodes_file = r'C:\Users\Hamed\Desktop\plates.csv'
node_coordinates_file = r'C:\Users\Hamed\Desktop\nodes.csv'
# Write element nodes to CSV
with open(element_nodes_file, mode='w', newline='') as elem_file:
    writer = csv.writer(elem_file)
    writer.writerow(['num', 'section', 'element_type', 'nodes'])
    for element in elements:
        # Get connectivity and reorder to anticlockwise (corner nodes followed by mid-side nodes)
        connectivity = element.connectivity
        anticlockwise_order = [
            connectivity[0],  # Node 0 (corner)
            connectivity[4],  # Node 4 (mid-side)
            connectivity[1],  # Node 1 (corner)
            connectivity[5],  # Node 5 (mid-side)
            connectivity[2],  # Node 2 (corner)
            connectivity[6],  # Node 6 (mid-side)
            connectivity[3],  # Node 3 (corner)
            connectivity[7],  # Node 7 (mid-side)
        ]
        node_string = "-".join(str(node) for node in anticlockwise_order)
        # Write element number and reordered node labels to the CSV
        writer.writerow([element.label - 1] + ["plate"] + ["Q8R"] + [node_string])
        print(f"Processed element {element.label} with nodes {anticlockwise_order}")
    print(f"Element nodes data successfully written to {element_nodes_file}")

# Write node coordinates to CSV
with open(node_coordinates_file, mode='w', newline='') as coord_file:
    writer = csv.writer(coord_file)
    writer.writerow(['num', 'x', 'y'])
    for node in nodes:
        writer.writerow([node.label - 1, node.coordinates[0], node.coordinates[1]])
        print(f"Processed node {node.label} with coordinates {node.coordinates}")
    print(f"Node coordinates data successfully written to {node_coordinates_file}")
