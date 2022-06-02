import os
import numpy as np
from src.settings import settings
from src.models import Node, Material, FrameSection, FrameElement2D, Structure, FrameYieldPoint

examples_dir = "input/examples/"
example_name = settings.example_name

global_cords_path = os.path.join(examples_dir, example_name, "global_cords.csv")
boundaries_path = os.path.join(examples_dir, example_name, "boundaries.csv")
joint_load_path = os.path.join(examples_dir, example_name, "loads/joint_loads.csv")
materials_path = os.path.join(examples_dir, example_name, "materials.csv")
sections_path = os.path.join(examples_dir, example_name, "elements/sections.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")
general_info_path = os.path.join(examples_dir, example_name, "general.csv")


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
    sections_array = np.loadtxt(fname=sections_path, usecols=range(8), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(sections_array.shape[0]):
        sections[sections_array[i, 0]] = FrameSection(
            material=materials[sections_array[i, 1]],
            a=float(sections_array[i, 2]),
            ix=float(sections_array[i, 3]),
            iy=float(sections_array[i, 4]),
            zp=float(sections_array[i, 5]),
            has_axial_yield=sections_array[i, 6],
            abar0=float(sections_array[i, 7]),
        )
    return sections


def create_frames():
    nodes = create_nodes()
    materials = create_materials()
    sections = create_sections(materials)
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    frames = []
    for i in range(frames_array.shape[0]):
        section = sections[frames_array[i, 0]]
        yield_point = FrameYieldPoint(section)
        frames.append(
            FrameElement2D(
                nodes=(nodes[int(frames_array[i, 1])], nodes[int(frames_array[i, 2])]),
                ends_fixity=frames_array[i, 3],
                section=section,
                yield_points=(yield_point, yield_point)
            )
        )
    return frames


def create_structure():
    boundaries_array = np.loadtxt(fname=boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    joint_loads = np.loadtxt(fname=joint_load_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    general_info = np.loadtxt(fname=general_info_path, usecols=range(3), delimiter=",", ndmin=1, skiprows=1, dtype=str)
    frames = create_frames()
    nodes_num = int(general_info[0])
    dim = general_info[1]
    load_limit = float(general_info[2])
    loads = {
        "joint_loads": joint_loads,
        "concentrated_member_loads": [],
        "distributed_load": [],
    }
    # FIXME: read nodes_num from input
    structure = Structure(
        nodes_num=nodes_num, dim=dim, elements=frames, boundaries=boundaries_array, loads=loads, load_limit=load_limit
    )
    return structure
