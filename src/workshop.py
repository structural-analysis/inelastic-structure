import os
import yaml
import logging
import numpy as np

from src.models.points import Node
from src.models.sections.frame import FrameSection
from src.models.sections.plate import PlateSection
from src.models.members.frame import FrameMember2D, Mass
from src.models.members.plate import PlateMember
from src.models.structure import Structure

examples_dir = "input/examples/"
general_file = "general.yaml"
global_cords_file = "global_cords.csv"
nodal_boundaries_file = "boundaries/nodal.csv"
linear_boundaries_file = "boundaries/linear.csv"
static_joint_loads_file = "loads/static/joint_loads.csv"
frame_sections_file = "sections/frames.yaml"
plate_sections_file = "sections/plates.yaml"
frame_members_file = "members/frames.csv"
plate_members_file = "members/plates.csv"
masses_file = "members/masses.csv"
load_limit_file = "limits/load.csv"
disp_limits_file = "limits/disp.csv"


def get_general_properties(example_name):
    general_file_path = os.path.join(examples_dir, example_name, general_file)
    with open(general_file_path, "r") as file_path:
        general_properties = yaml.safe_load(file_path)
    return general_properties


def create_nodes(example_name, structure_dim):
    nodes = []
    global_cords_path = os.path.join(examples_dir, example_name, global_cords_file)
    if structure_dim.lower() == "2d":
        nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
        for i in range(nodes_array.shape[0]):
            nodes.append(Node(
                num=i,
                x=nodes_array[i][0],
                y=nodes_array[i][1],
                z=0,
            ))
    elif structure_dim.lower() == "3d":
        nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1)
        for i in range(nodes_array.shape[0]):
            nodes.append(Node(
                num=i,
                x=nodes_array[i][0],
                y=nodes_array[i][1],
                z=nodes_array[i][2],
            ))
    return nodes


def create_frame_sections(example_name):
    frame_sections = {}
    frame_sections_path = os.path.join(examples_dir, example_name, frame_sections_file)
    try:
        with open(frame_sections_path, "r") as path:
            frame_sections_dict = yaml.safe_load(path)
            for key, value in frame_sections_dict.items():
                frame_sections[key] = FrameSection(input=value)
            return frame_sections
    except FileNotFoundError:
        logging.warning("frame sections input file not found")
        return {}


def create_plate_sections(example_name):
    plate_sections = {}
    plate_sections_path = os.path.join(examples_dir, example_name, plate_sections_file)
    try:
        with open(plate_sections_path, "r") as path:
            plate_sections_dict = yaml.safe_load(path)
            for key, value in plate_sections_dict.items():
                plate_sections[key] = PlateSection(input=value)
            return plate_sections
    except FileNotFoundError:
        logging.warning("plate sections input file not found")
        return {}


def create_frame_masses(example_name):
    # mass per length is applied in global direction so there is no need to transform.
    frame_masses = {}
    masses_path = os.path.join(examples_dir, example_name, masses_file)
    try:
        masses_array = np.loadtxt(fname=masses_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    except FileNotFoundError:
        logging.warning("mass input file not found")
        return {}
    for i in range(masses_array.shape[0]):
        frame_masses[int(masses_array[i][0])] = masses_array[i][1]
    return frame_masses


def create_frame_members(example_name, nodes, general_properties):
    frame_members_path = os.path.join(examples_dir, example_name, frame_members_file)
    frame_sections = create_frame_sections(example_name)
    frame_masses = create_frame_masses(example_name) if general_properties.get("dynamic_analysis") else {}
    frame_members = []

    try:
        frames_array = np.loadtxt(fname=frame_members_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("frame members input file not found")
        return []

    if frame_sections:
        for i in range(frames_array.shape[0]):
            frame_section = frame_sections[frames_array[i, 0]]
            frame_members.append(
                FrameMember2D(
                    nodes=(nodes[int(frames_array[i, 1])], nodes[int(frames_array[i, 2])]),
                    ends_fixity=frames_array[i, 3],
                    section=frame_section,
                    mass=Mass(magnitude=frame_masses.get(i)) if frame_masses.get(i) else None
                )
            )
    return frame_members


def create_plate_members(example_name, nodes):
    plate_members_path = os.path.join(examples_dir, example_name, plate_members_file)
    plate_sections = create_plate_sections(example_name)
    plate_members = []

    try:
        plates_array = np.loadtxt(fname=plate_members_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("plate members input file not found")
        return []

    if plate_sections:
        for i in range(plates_array.shape[0]):
            plate_section = plate_sections[plates_array[i, 0]]
            plate_members.append(
                PlateMember(
                    section=plate_section,
                    nodes=(
                        nodes[int(plates_array[i, 1][0])],
                        nodes[int(plates_array[i, 1][1])],
                        nodes[int(plates_array[i, 1][2])],
                        nodes[int(plates_array[i, 1][3])],
                    ),
                    mesh_num=(int(plates_array[i, 2]), int(plates_array[i, 3])),
                )
            )
    return plate_members


def create_structure(example_name):
    nodal_boundaries_path = os.path.join(examples_dir, example_name, nodal_boundaries_file)
    linear_boundaries_path = os.path.join(examples_dir, example_name, linear_boundaries_file)
    joint_load_path = os.path.join(examples_dir, example_name, static_joint_loads_file)
    load_limit_path = os.path.join(examples_dir, example_name, load_limit_file)
    disp_limits_path = os.path.join(examples_dir, example_name, disp_limits_file)

    try:
        nodal_boundaries = np.loadtxt(fname=nodal_boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    except FileNotFoundError:
        nodal_boundaries = None
    try:
        linear_boundaries = np.loadtxt(fname=linear_boundaries_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=int)
    except FileNotFoundError:
        linear_boundaries = None

    joint_loads = np.loadtxt(fname=joint_load_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    load_limit = np.loadtxt(fname=load_limit_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=float)
    disp_limits = np.loadtxt(fname=disp_limits_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)

    general_properties = get_general_properties(example_name)
    nodes = create_nodes(example_name, structure_dim=general_properties["structure_dim"])

    frame_members = create_frame_members(
        example_name=example_name,
        general_properties=general_properties,
        nodes=nodes,
    )

    plate_members = create_plate_members(
        example_name=example_name,
        nodes=nodes,
    )

    limits = {
        "load_limit": load_limit,
        "disp_limits": disp_limits
    }

    loads = {
        "joint_loads": joint_loads,
        "concentrated_member_loads": [],
        "distributed_load": [],
    }

    input = {
        "nodes_num": len(nodes),
        "dim": general_properties["structure_dim"],
        "include_softening": general_properties["include_softening"],
        "members": frame_members + plate_members,
        "nodal_boundaries": nodal_boundaries,
        "linear_boundaries": linear_boundaries,
        "loads": loads,
        "limits": limits,
    }

    structure = Structure(
        input=input,
    )

    return structure
