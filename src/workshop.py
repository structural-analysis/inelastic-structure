import os
import numpy as np
from src.settings import settings
from src.models import Node, Material, FrameSection, FrameElement2D, Structure

examples_dir = "input/examples/"
example_name = settings.example_name

global_cords_path = os.path.join(examples_dir, example_name, "global_cords.csv")
boundaries_path = os.path.join(examples_dir, example_name, "boundaries.csv")
joint_load_path = os.path.join(examples_dir, example_name, "loads/joint_loads.csv")
materials_path = os.path.join(examples_dir, example_name, "materials.csv")
sections_path = os.path.join(examples_dir, example_name, "elements/sections.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")


def create_nodes():
    nodes = []
    nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(nodes_array.shape[0]):
        x = nodes_array[i][0]
        y = nodes_array[i][1]
        nodes.append(Node(num=i, x=x, y=y))
    return nodes


def create_materials():
    materials = {}
    materials_array = np.loadtxt(fname=materials_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(materials_array.shape[0]):
        materials[materials_array[i, 0]] = Material(name=materials_array[i, 0])
    return materials


def create_sections(materials):
    sections = {}
    sections_array = np.loadtxt(fname=sections_path, usecols=range(7), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(sections_array.shape[0]):
        sections[sections_array[i, 0]] = FrameSection(
            material=materials[sections_array[i, 1]],
            a=float(sections_array[i, 2]),
            ix=float(sections_array[i, 3]),
            iy=float(sections_array[i, 4]),
            zp=float(sections_array[i, 5]),
            has_axial=bool(sections_array[i, 6]),
        )
    return sections


def create_frames():
    nodes = create_nodes()
    materials = create_materials()
    sections = create_sections(materials)
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    frames = []
    for i in range(frames_array.shape[0]):
        frames.append(
            FrameElement2D(
                nodes=(nodes[int(frames_array[i, 1])], nodes[int(frames_array[i, 2])]),
                section=sections[frames_array[i, 0]],
                ends_fixity=frames_array[i, 3],
            )
        )
    return frames


def create_structure():
    boundaries_array = np.loadtxt(fname=boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    frames = create_frames()
    joint_loads = np.loadtxt(fname=joint_load_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    loads = {
        "joint_loads": joint_loads,
        "concentrated_member_loads": [],
        "distributed_load": [],
    }
    structure = Structure(
        n_nodes=3, node_n_dof=3, elements=frames, boundaries=boundaries_array, loads=loads
    )
    return structure
