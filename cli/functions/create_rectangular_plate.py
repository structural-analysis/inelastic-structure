import os
import numpy as np
from dataclasses import dataclass

from src.models.points import Node


@dataclass
class Boundaries:
    corners: list
    bottoms: list
    rights: list
    tops: list
    lefts: list


def is_odd_or_even(number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"


def tuple_to_dash_string(numbers_tuple):
    # Convert each number in the tuple to a string
    str_numbers = map(str, numbers_tuple)
    # Join the string representations of the numbers with a dash
    result_string = '-'.join(str_numbers)
    return result_string


def get_boundaries(nodes, xsize, ysize):
    corners = []
    bottoms = []
    rights = []
    tops = []
    lefts = []

    for node in nodes:
        if node.x == 0:
            if node.y != 0 and node.y != ysize:
                lefts.append(node)
        if node.x == xsize:
            if node.y != 0 and node.y != ysize:
                rights.append(node)
        if node.y == 0:
            if node.x != 0 and node.x != xsize:
                bottoms.append(node)
        if node.y == ysize:
            if node.x != 0 and node.x != xsize:
                tops.append(node)

        if node.x == 0 and node.y == 0:
            corners.append(node)
        if node.x == xsize and node.y == 0:
            corners.append(node)
        if node.x == xsize and node.y == ysize:
            corners.append(node)
        if node.x == 0 and node.y == ysize:
            corners.append(node)
    return Boundaries(
        corners=corners,
        bottoms=bottoms,
        rights=rights,
        tops=tops,
        lefts=lefts,
    )


def generate_mesh(xsize, ysize, xnum, ynum):
    dx = xsize / (2 * xnum)
    dy = ysize / (2 * ynum)

    nodes = []
    k = 0
    for j in range(2 * ynum + 1):
        for i in range(2 * xnum + 1):
            if is_odd_or_even(i) == "Odd" and is_odd_or_even(j) == "Odd":
                continue
            else:
                nodes.append(Node(num=k, x=i*dx, y=j*dy, z=0))
            k += 1

    members = []
    for j in range(ynum):
        for i in range(xnum):
            n0 = (4 * xnum - 1) * j + 2 * i
            n1 = n0 + 1
            n2 = n0 + 2
            n3 = (4 * xnum - 1) * j + 2 * xnum + 2 + i
            n4 = (4 * xnum - 1) * j + (4 * xnum + 1) + 2 * i
            n5 = n4 - 1
            n6 = n4 - 2
            n7 = n3 - 1
            
            members.append((n0, n1, n2, n3, n4, n5, n6, n7))
    
    return nodes, members


def write_nodes_to_csv(nodes, filename='nodes.csv'):
    nodes_array = np.array([[node.num, node.x, node.y] for node in nodes])
    np.savetxt(filename, nodes_array, delimiter=',', header='num,x,y', comments='', fmt='%d,%.2f,%.2f')


def write_members_to_csv(members, filename='members.csv'):
    members_array = np.array([[i, 'plate', 'Q8R', tuple_to_dash_string(nodes)] for i, nodes in enumerate(members)], dtype=object)
    header = 'num,section,member_type,nodes'
    np.savetxt(filename, members_array, delimiter=',', header=header, comments='', fmt='%s')


def write_boundaries_to_csv(boundaries, filename='boundaries.csv'):
    boundaries_list = []
    for corner in boundaries.corners:
        boundaries_list.append((corner.num, 0))
    for bottom in boundaries.bottoms:
        boundaries_list.append((bottom.num, 0))
        boundaries_list.append((bottom.num, 2))
    for right in boundaries.rights:
        boundaries_list.append((right.num, 0))
        boundaries_list.append((right.num, 1))
    for top in boundaries.tops:
        boundaries_list.append((top.num, 0))
        boundaries_list.append((top.num, 2))
    for left in boundaries.lefts:
        boundaries_list.append((left.num, 0))
        boundaries_list.append((left.num, 1))

    boundaries_array = np.array([[boundary[0], boundary[1]] for boundary in boundaries_list])
    np.savetxt(filename, boundaries_array, delimiter=',', header='node,dof', comments='', fmt='%d,%d')


def create_example(name, xsize, ysize, xnum, ynum):
    nodes, members = generate_mesh(xsize, ysize, xnum, ynum)
    boundaries = get_boundaries(nodes, xsize, ysize)

    example_path = f"input/examples/{name}"
    memebers_path = f"{example_path}/members"
    boundaries_path = f"{example_path}/boundaries"

    os.makedirs(example_path, exist_ok=True)
    os.makedirs(memebers_path, exist_ok=True)
    os.makedirs(boundaries_path, exist_ok=True)

    write_nodes_to_csv(nodes, f"{example_path}/nodes.csv")
    write_members_to_csv(members, f"{example_path}/members/plates.csv")
    write_boundaries_to_csv(boundaries, f"{example_path}/boundaries/nodal.csv")
