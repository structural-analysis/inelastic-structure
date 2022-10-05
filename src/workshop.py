import os
import yaml
import numpy as np

from src.models.points import Node
from src.models.sections.frame import FrameSection
from src.models.elements.frame import FrameElement2D, Mass
from src.models.loads import Dynamic, Joint, Loads
from src.models.structure import Structure

examples_dir = "input/examples/"
global_cords_dir = "global_cords.csv"
boundaries_dir = "boundaries.csv"
joint_loads_dir = "loads/static/joint_loads.csv"
sections_dir = "sections"
frames_dir = "members/frames.csv"
masses_dir = "members/masses.csv"
general_dir = "general.yaml"
load_limit_dir = "limits/load.csv"
disp_limits_dir = "limits/disp.csv"
dynamic_loads_dir = "loads/dynamic"


def create_nodes(example_name):
    nodes = []
    global_cords_path = os.path.join(examples_dir, example_name, global_cords_dir)
    nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(nodes_array.shape[0]):
        x = nodes_array[i][0]
        y = nodes_array[i][1]
        nodes.append(Node(num=i, x=x, y=y))
    return nodes


def create_sections(example_name):
    sections = {}
    sections_path = os.path.join(examples_dir, example_name, sections_dir)
    for section_path in os.scandir(sections_path):
        with open(section_path, "r") as section_file:
            section = yaml.safe_load(section_file)
            sections[section["name"]] = FrameSection(
                input=section["input"],
            )
    return sections


def create_masses(example_name):
    # mass per length is applied in global direction so there is no need to transform.
    masses = {}
    masses_path = os.path.join(examples_dir, example_name, masses_dir)
    masses_array = np.loadtxt(fname=masses_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    for i in range(masses_array.shape[0]):
        masses[int(masses_array[i][0])] = masses_array[i][1]
    return masses


def create_frames(example_name):
    general_info = get_general_info(example_name)
    frames_path = os.path.join(examples_dir, example_name, frames_dir)
    nodes = create_nodes(example_name)
    sections = create_sections(example_name)
    masses = create_masses(example_name) if general_info.get("dynamic_analysis") else {}
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    frames = []
    for i in range(frames_array.shape[0]):
        section = sections[frames_array[i, 0]]
        frames.append(
            FrameElement2D(
                nodes=(nodes[int(frames_array[i, 1])], nodes[int(frames_array[i, 2])]),
                ends_fixity=frames_array[i, 3],
                section=section,
                mass=Mass(magnitude=masses.get(i)) if masses.get(i) else None
            )
        )
    return frames


def create_joint_load(example_name):
    general_info = get_general_info(example_name)
    if not general_info.get("dynamic_analysis"):
        joint_loads_path = os.path.join(examples_dir, example_name, joint_loads_dir)
        joint_loads__array = np.loadtxt(fname=joint_loads_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
        joint_loads = []
        for i in range(joint_loads__array.shape[0]):
            joint_loads.append(
                Joint(
                    node=int(joint_loads__array[i, 0]),
                    dof=int(joint_loads__array[i, 1]),
                    magnitude=joint_loads__array[i, 2],
                )
            )
        return joint_loads
    else:
        return []


def create_dynamic_loads(example_name):
    general_info = get_general_info(example_name)
    if general_info.get("dynamic_analysis"):
        dynamic_joint_loads_dir = f"{dynamic_loads_dir}/joint_loads.csv"
        dynamic_loads_time_dir = f"{dynamic_loads_dir}/time.csv"
        dynamic_loads_time_path = os.path.join(examples_dir, example_name, dynamic_loads_time_dir)
        dynamic_joint_loads_path = os.path.join(examples_dir, example_name, dynamic_joint_loads_dir)

        dynamic_joint_loads_array = np.loadtxt(fname=dynamic_joint_loads_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
        time = np.loadtxt(fname=dynamic_loads_time_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=1, dtype=float)
        dynamic_loads = []
        for i in range(dynamic_joint_loads_array.shape[0]):
            dynamic_load_dir = f"{dynamic_loads_dir}/{dynamic_joint_loads_array[i, 0]}.csv"
            dynamic_load_path = os.path.join(examples_dir, example_name, dynamic_load_dir)
            load = np.loadtxt(fname=dynamic_load_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=0, dtype=float)
            dynamic_loads.append(
                Dynamic(
                    node=int(dynamic_joint_loads_array[i, 1]),
                    dof=int(dynamic_joint_loads_array[i, 2]),
                    time=np.matrix(time),
                    magnitude=np.matrix(load * float(dynamic_joint_loads_array[i, 3])),
                )
            )
        return dynamic_loads
    else:
        return []


def get_structure_input(example_name):
    boundaries_path = os.path.join(examples_dir, example_name, boundaries_dir)
    general_info = get_general_info(example_name)
    load_limit_path = os.path.join(examples_dir, example_name, load_limit_dir)
    disp_limits_path = os.path.join(examples_dir, example_name, disp_limits_dir)

    boundaries = np.loadtxt(fname=boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    load_limit = np.loadtxt(fname=load_limit_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=float)
    disp_limits = np.loadtxt(fname=disp_limits_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)

    elements_list = create_frames(example_name)
    dynamic_loads = create_dynamic_loads(example_name)
    joint_loads = create_joint_load(example_name)

    limits = {
        "load_limit": load_limit,
        "disp_limits": disp_limits
    }

    loads = {
        "joint": joint_loads,
        "concentrated": [],
        "distributed": [],
        "dynamic": dynamic_loads,
    }

    input = {
        "general": general_info,
        "elements_list": elements_list,
        "boundaries": boundaries,
        "loads": loads,
        "limits": limits,
    }
    return input


def get_loads_input(example_name):
    dynamic_loads = create_dynamic_loads(example_name)
    joint_loads = create_joint_load(example_name)
    input = {
        "joint": joint_loads,
        "concentrated": [],
        "distributed": [],
        "dynamic": dynamic_loads,
    }
    return input


def get_general_info(example_name):
    general_info_path = os.path.join(examples_dir, example_name, general_dir)
    with open(general_info_path, "r") as general_file:
        general_info = yaml.safe_load(general_file)
    return general_info
