import numpy as np

def is_odd_or_even(number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"

def generate_mesh(a, b, m, n):
    dx = a / (2 * m)
    dy = b / (2 * n)

    nodes = []
    for j in range(2 * n + 1):
        for i in range(2 * m + 1):
            if is_odd_or_even(i) == "Odd" and is_odd_or_even(j) == "Odd":
                continue
            else:
                nodes.append((i * dx, j * dy))

    elements = []
    for j in range(n):
        for i in range(m):
            n0 = (4 * m - 1) * j + 2 * i
            n1 = n0 + 1
            n2 = n0 + 2
            n3 = (4 * m - 1) * j + 2 * m + 2 + i
            n4 = (4 * m - 1) * j + (4 * m + 1) + 2 * i
            n5 = n4 - 1
            n6 = n4 - 2
            n7 = n3 - 1
            
            elements.append((n0, n1, n2, n3, n4, n5, n6, n7))
    
    return nodes, elements

def write_nodes_to_csv(nodes, filename='nodes.csv'):
    nodes_array = np.array([[i, x, y] for i, (x, y) in enumerate(nodes)])
    np.savetxt(filename, nodes_array, delimiter=',', header='num,x,y', comments='', fmt='%d,%.2f,%.2f')

def write_elements_to_csv(elements, filename='elements.csv'):
    elements_array = np.array([[i, 'plate', 'Q8R'] + list(nodes) for i, nodes in enumerate(elements)], dtype=object)
    header = 'num,section,element_type,' + ','.join([f'node{i}' for i in range(8)])
    np.savetxt(filename, elements_array, delimiter=',', header=header, comments='', fmt='%s')

# Example usage
a = 6
b = 3
m = 3
n = 2

nodes, elements = generate_mesh(a, b, m, n)
write_nodes_to_csv(nodes, 'nodes.csv')
write_elements_to_csv(elements, 'elements.csv')
