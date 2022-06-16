import os
import numpy as np
from src.models.points import Node, FrameYieldPoint
from src.models.materials import Material
from src.models.sections import FrameSection
from src.models.elements import FrameElement2D
from src.models.structure import Structure

examples_dir = "input/examples/"
global_cords_dir = "global_cords.csv"
boundaries_dir = "boundaries.csv"
joint_loads_dir = "loads/joint_loads.csv"
materials_dir = "materials.csv"
sections_dir = "elements/sections.csv"
frames_dir = "members/frames.csv"
general_dir = "general.csv"
load_limit_dir = "limits/load.csv"
disp_limits_dir = "limits/disp.csv"


def create_nodes(example_name):
    nodes = []
    global_cords_path = os.path.join(examples_dir, example_name, global_cords_dir)
    nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(nodes_array.shape[0]):
        x = nodes_array[i][0]
        y = nodes_array[i][1]
        nodes.append(Node(num=i, x=x, y=y))
    return nodes


def create_materials(example_name):
    materials = {}
    materials_path = os.path.join(examples_dir, example_name, materials_dir)
    materials_array = np.loadtxt(fname=materials_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(materials_array.shape[0]):
        materials[materials_array[i, 0]] = Material(name=materials_array[i, 0])
    return materials


def create_sections(materials, example_name):
    sections = {}
    sections_path = os.path.join(examples_dir, example_name, sections_dir)
    sections_array = np.loadtxt(fname=sections_path, usecols=range(15), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(sections_array.shape[0]):
        sections[sections_array[i, 0]] = FrameSection(
            material=materials[sections_array[i, 1]],
            a=float(sections_array[i, 2]),
            ix=float(sections_array[i, 3]),
            iy=float(sections_array[i, 4]),
            zp=float(sections_array[i, 5]),
            has_axial_yield=sections_array[i, 6],
            abar0=float(sections_array[i, 7]),
            ap=float(sections_array[i, 8]),
            mp=float(sections_array[i, 9]),
            is_direct_capacity=sections_array[i, 10],
            include_softening=sections_array[i, 11],
            alpha=float(sections_array[i, 12]),
            ep1=float(sections_array[i, 13]),
            ep2=float(sections_array[i, 14]),
        )
    return sections


def create_frames(example_name):
    frames_path = os.path.join(examples_dir, example_name, frames_dir)
    nodes = create_nodes(example_name)
    materials = create_materials(example_name)
    sections = create_sections(materials, example_name)
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


def create_structure(example_name):
    boundaries_path = os.path.join(examples_dir, example_name, boundaries_dir)
    joint_load_path = os.path.join(examples_dir, example_name, joint_loads_dir)
    general_info_path = os.path.join(examples_dir, example_name, general_dir)
    load_limit_path = os.path.join(examples_dir, example_name, load_limit_dir)
    disp_limits_path = os.path.join(examples_dir, example_name, disp_limits_dir)

    boundaries = np.loadtxt(fname=boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    joint_loads = np.loadtxt(fname=joint_load_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    general_info = np.loadtxt(fname=general_info_path, usecols=range(3), delimiter=",", ndmin=1, skiprows=1, dtype=str)
    load_limit = np.loadtxt(fname=load_limit_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=float)
    disp_limits = np.loadtxt(fname=disp_limits_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)

    frames = create_frames(example_name)
    nodes_num = int(general_info[0])
    dim = general_info[1]

    limits = {
        "load_limit": load_limit,
        "disp_limits": disp_limits
    }

    loads = {
        "joint_loads": joint_loads,
        "concentrated_member_loads": [],
        "distributed_load": [],
    }

    structure = Structure(
        nodes_num=nodes_num,
        dim=dim,
        elements=frames,
        boundaries=boundaries,
        loads=loads,
        limits=limits,
    )

    return structure
