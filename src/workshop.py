import os
import yaml
import numpy as np
from src.models.points import Node
from src.models.materials import Material
from src.models.sections import FrameSection
from src.models.elements import Elements, FrameElement2D
from src.models.structure import Structure

examples_dir = "input/examples/"
global_cords_dir = "global_cords.csv"
boundaries_dir = "boundaries.csv"
joint_loads_dir = "loads/joint_loads.csv"
materials_dir = "materials.csv"
sections_dir = "sections"
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
    for section_path in os.scandir(sections_path):
        with open(section_path, "r") as section_file:
            section = yaml.safe_load(section_file)
            sections[section["name"]] = FrameSection(
                material=materials[section["material"]],
                a=float(section["a"]),
                ix=float(section["ix"]),
                iy=float(section["iy"]),
                nonlinear=section["nonlinear"],
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
        frames.append(
            FrameElement2D(
                nodes=(nodes[int(frames_array[i, 1])], nodes[int(frames_array[i, 2])]),
                ends_fixity=frames_array[i, 3],
                section=section,
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

    elements_array = create_frames(example_name)
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
        elements=Elements(elements_array),
        boundaries=boundaries,
        loads=loads,
        limits=limits,
    )

    return structure
