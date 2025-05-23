import os
import yaml
import logging
import numpy as np
from enum import Enum

from src.models.points import Node
from src.models.boundaries import NodalBoundary, LinearBoundary
from src.models.sections.truss2d import Truss2DSection
from src.models.sections.frame2d import Frame2DSection
from src.models.sections.frame3d import Frame3DSection
from src.models.sections.plate import PlateSection
from src.models.sections.wall import WallSection
from src.models.members.truss2d import Truss2DMember
from src.models.members.frame2d import Frame2DMember, Mass
from src.models.members.frame3d import Frame3DMember
from src.models.members.plate import PlateMember
from src.models.members.wall import WallMember
from src.models.loads import DynamicLoad, JointLoad, DistributedLoad

examples_dir = "input/examples/"
general_file = "general.yaml"
nodes_file = "nodes.csv"
nodal_boundaries_file = "boundaries/nodal.csv"
linear_boundaries_file = "boundaries/linear.csv"
truss2d_sections_file = "sections/trusses2d.yaml"
frame2d_sections_file = "sections/frames2d.yaml"
frame3d_sections_file = "sections/frames3d.yaml"
plate_sections_file = "sections/plates.yaml"
wall_sections_file = "sections/walls.yaml"
truss2d_members_file = "members/trusses2d.csv"
frame2d_members_file = "members/frames2d.csv"
frame3d_members_file = "members/frames3d.csv"
plate_members_file = "members/plates.csv"
wall_members_file = "members/walls.csv"
masses_file = "members/masses.csv"
load_limits_file = "limits/load.csv"
disp_limits_file = "limits/disp.csv"
dynamic_loads_dir = "loads/dynamic/"
joint_loads_file = "loads/static/joint_loads.csv"
distributed_load_file = "loads/static/distributed_load.csv"
output_dir = "output/examples/"


class StructureNodeDOF(int, Enum):
    TRUSS2D = 2
    FRAME2D = 3
    FRAME3D = 6
    WALL2D = 2
    PLATE2D = 3


def get_general_properties(example_name):
    general_file_path = os.path.join(examples_dir, example_name, general_file)
    with open(general_file_path, "r") as file_path:
        general_properties = yaml.safe_load(file_path)
    return general_properties


def create_initial_nodes(example_name, structure_dim):
    initial_nodes = []
    nodes_path = os.path.join(examples_dir, example_name, nodes_file)
    if structure_dim.lower() == "2d":
        nodes_array = np.loadtxt(fname=nodes_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1)
        for i in range(nodes_array.shape[0]):
            initial_nodes.append(Node(
                num=int(nodes_array[i][0]),
                x=nodes_array[i][1],
                y=nodes_array[i][2],
                z=0,
            ))
    elif structure_dim.lower() == "3d":
        nodes_array = np.loadtxt(fname=nodes_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1)
        for i in range(nodes_array.shape[0]):
            initial_nodes.append(Node(
                num=int(nodes_array[i][0]),
                x=nodes_array[i][1],
                y=nodes_array[i][2],
                z=nodes_array[i][3],
            ))
    return initial_nodes


def create_nodal_boundaries(example_name, initial_nodes):
    nodal_boundaries = []
    nodal_boundaries_path = os.path.join(examples_dir, example_name, nodal_boundaries_file)
    try:
        nodal_boundaries_array = np.loadtxt(fname=nodal_boundaries_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=int)
        for i in range(nodal_boundaries_array.shape[0]):
            nodal_boundaries.append(NodalBoundary(
                node=initial_nodes[nodal_boundaries_array[i][0]],
                dof=nodal_boundaries_array[i][1],
            ))
    except FileNotFoundError:
        pass
    return nodal_boundaries


def create_linear_boundaries(example_name, initial_nodes):
    linear_boundaries = []
    linear_boundaries_path = os.path.join(examples_dir, example_name, linear_boundaries_file)
    try:
        linear_boundaries_array = np.loadtxt(fname=linear_boundaries_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=int)
        for i in range(linear_boundaries_array.shape[0]):
            linear_boundaries.append(LinearBoundary(
                start_node=initial_nodes[linear_boundaries_array[i][0]],
                end_node=initial_nodes[linear_boundaries_array[i][1]],
                dof=linear_boundaries_array[i][2],
            ))
    except FileNotFoundError:
        pass
    return linear_boundaries


def create_truss2d_sections(example_name, general_properties):
    truss2d_sections = {}
    truss2d_sections_path = os.path.join(examples_dir, example_name, truss2d_sections_file)
    is_inelastic = general_properties["inelastic"]["enabled"]
    try:
        with open(truss2d_sections_path, "r") as path:
            truss2d_sections_dict = yaml.safe_load(path)

        if is_inelastic:
            nonlinear_capacity_dir = f"{output_dir}/{example_name}/nonlinear_capacity"
            if not os.path.exists(nonlinear_capacity_dir):
                os.makedirs(nonlinear_capacity_dir)

        for key, value in truss2d_sections_dict.items():
            truss2d_sections[key] = Truss2DSection(input=value)
            if is_inelastic:
                with open(f"{nonlinear_capacity_dir}/{key}.csv", "w") as ff:
                    ff.write(f"0,ap_positive,{truss2d_sections[key].nonlinear.ap_positive}\n")
                    ff.write(f"0,ap_negative,{truss2d_sections[key].nonlinear.ap_negative}\n")

        return truss2d_sections
    except FileNotFoundError:
        logging.warning("truss2d sections input file not found")
        return {}


def create_frame2d_sections(example_name, general_properties):
    frame_sections = {}
    frame_sections_path = os.path.join(examples_dir, example_name, frame2d_sections_file)
    is_inelastic = general_properties["inelastic"]["enabled"]
    try:
        with open(frame_sections_path, "r") as path:
            frame_sections_dict = yaml.safe_load(path)

        if is_inelastic:
            nonlinear_capacity_dir = f"{output_dir}/{example_name}/nonlinear_capacity"
            if not os.path.exists(nonlinear_capacity_dir):
                os.makedirs(nonlinear_capacity_dir)

        for key, value in frame_sections_dict.items():
            frame_sections[key] = Frame2DSection(input=value)
            if is_inelastic:
                with open(f"{nonlinear_capacity_dir}/{key}.csv", "w") as ff:
                    ff.write(f"0,ap,{frame_sections[key].nonlinear.ap}\n")
                    ff.write(f"2,mp,{frame_sections[key].nonlinear.mp}\n")

        return frame_sections
    except FileNotFoundError:
        logging.warning("frame sections input file not found")
        return {}


def create_frame3d_sections(example_name, general_properties):
    frame_sections = {}
    frame_sections_path = os.path.join(examples_dir, example_name, frame3d_sections_file)
    is_inelastic = general_properties["inelastic"]["enabled"]
    try:
        with open(frame_sections_path, "r") as path:
            frame_sections_dict = yaml.safe_load(path)

        if is_inelastic:
            nonlinear_capacity_dir = f"{output_dir}/{example_name}/nonlinear_capacity"
            if not os.path.exists(nonlinear_capacity_dir):
                os.makedirs(nonlinear_capacity_dir)

        for key, value in frame_sections_dict.items():
            frame_sections[key] = Frame3DSection(input=value)
            if is_inelastic:
                with open(f"{nonlinear_capacity_dir}/{key}.csv", "w") as ff:
                    ff.write(f"0,ap,{frame_sections[key].nonlinear.ap}\n")
                    ff.write(f"4,mp22,{frame_sections[key].nonlinear.mp22}\n")
                    ff.write(f"5,mp33,{frame_sections[key].nonlinear.mp33}\n")

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


def create_wall_sections(example_name):
    wall_sections = {}
    wall_sections_path = os.path.join(examples_dir, example_name, wall_sections_file)
    try:
        with open(wall_sections_path, "r") as path:
            wall_sections_dict = yaml.safe_load(path)
            for key, value in wall_sections_dict.items():
                wall_sections[key] = WallSection(input=value)
            return wall_sections
    except FileNotFoundError:
        logging.warning("wall sections input file not found")
        return {}


def create_truss2d_masses(example_name):
    # mass per length is applied in global direction so there is no need to transform.
    truss2d_masses = {}
    masses_path = os.path.join(examples_dir, example_name, masses_file)
    try:
        masses_array = np.loadtxt(fname=masses_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    except FileNotFoundError:
        logging.warning("mass input file not found")
        return {}
    for i in range(masses_array.shape[0]):
        truss2d_masses[int(masses_array[i][0])] = masses_array[i][1]
    return truss2d_masses


def create_frame2d_masses(example_name):
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


def create_frame3d_masses(example_name):
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


def create_truss2d_members(example_name, node_dofs_count, nodes, general_properties, include_softening):
    truss2d_members_path = os.path.join(examples_dir, example_name, truss2d_members_file)
    truss2d_sections = create_truss2d_sections(example_name, general_properties)
    truss2d_masses = create_truss2d_masses(example_name) if general_properties.get("dynamic_analysis") else {}
    truss2d_members = []

    try:
        truss2ds_array = np.loadtxt(fname=truss2d_members_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("truss2d members input file not found")
        return []

    if truss2d_sections:
        for i in range(truss2ds_array.shape[0]):
            member_num = int(truss2ds_array[i, 0])
            truss2d_section = truss2d_sections[truss2ds_array[i, 1]]
            member_nodes = truss2ds_array[i, 2]
            split_nodes = member_nodes.split("-")
            truss2d_members.append(
                Truss2DMember(
                    num=member_num,
                    section=truss2d_section,
                    include_softening=include_softening,
                    nodes=(
                        nodes[int(split_nodes[0])],
                        nodes[int(split_nodes[1])],
                    ),
                    mass=Mass(magnitude=truss2d_masses.get(i)) if truss2d_masses.get(i) else None,
                )
            )
    return truss2d_members


def create_frame2d_members(example_name, node_dofs_count, nodes, general_properties, include_softening):
    frame_members_path = os.path.join(examples_dir, example_name, frame2d_members_file)
    frame_sections = create_frame2d_sections(example_name, general_properties)
    frame_masses = create_frame2d_masses(example_name) if general_properties.get("dynamic_analysis") else {}
    frame_members = []

    try:
        frames_array = np.loadtxt(fname=frame_members_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("frame members input file not found")
        return []

    if frame_sections:
        for i in range(frames_array.shape[0]):
            member_num = int(frames_array[i, 0])
            frame_section = frame_sections[frames_array[i, 1]]
            member_nodes = frames_array[i, 2]
            split_nodes = member_nodes.split("-")
            frame_members.append(
                Frame2DMember(
                    num=member_num,
                    section=frame_section,
                    include_softening=include_softening,
                    nodes=(
                        nodes[int(split_nodes[0])],
                        nodes[int(split_nodes[1])],
                    ),
                    mass=Mass(magnitude=frame_masses.get(i)) if frame_masses.get(i) else None,
                    ends_fixity=frames_array[i, 3],
                )
            )
    return frame_members


def create_frame3d_members(example_name, node_dofs_count, nodes, general_properties, include_softening):
    frame_members_path = os.path.join(examples_dir, example_name, frame3d_members_file)
    frame_sections = create_frame3d_sections(example_name, general_properties)
    frame_masses = create_frame3d_masses(example_name) if general_properties.get("dynamic_analysis") else {}
    frame_members = []

    try:
        frames_array = np.loadtxt(fname=frame_members_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("frame members input file not found")
        return []

    if frame_sections:
        for i in range(frames_array.shape[0]):
            member_num = int(frames_array[i, 0])
            frame_section = frame_sections[frames_array[i, 1]]
            member_nodes = frames_array[i, 2]
            split_nodes = member_nodes.split("-")
            frame_members.append(
                Frame3DMember(
                    num=member_num,
                    section=frame_section,
                    include_softening=include_softening,
                    nodes=(
                        nodes[int(split_nodes[0])],
                        nodes[int(split_nodes[1])],
                    ),
                    mass=Mass(magnitude=frame_masses.get(i)) if frame_masses.get(i) else None,
                    ends_fixity=frames_array[i, 3],
                )
            )
    return frame_members


def create_plate_members(example_name, nodes, node_dofs_count, include_softening):
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
            member_num = int(plates_array[i, 0])
            plate_section = plate_sections[plates_array[i, 1]]
            element_type = plates_array[i, 2].upper()
            member_nodes = plates_array[i, 3]
            split_nodes = member_nodes.split("-")
            final_nodes = [nodes[int(split_node)] for split_node in split_nodes]
            plate_members.append(
                PlateMember(
                    num=member_num,
                    section=plate_section,
                    include_softening=include_softening,
                    element_type=element_type,
                    nodes=tuple(final_nodes)
                )
            )
    return plate_members


def create_wall_members(example_name, nodes, node_dofs_count, include_softening):
    wall_members_path = os.path.join(examples_dir, example_name, wall_members_file)
    wall_sections = create_wall_sections(example_name)
    wall_members = []

    try:
        walls_array = np.loadtxt(fname=wall_members_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    except FileNotFoundError:
        logging.warning("wall members input file not found")
        return []

    if wall_sections:
        for i in range(walls_array.shape[0]):
            member_num = int(walls_array[i, 0])
            wall_section = wall_sections[walls_array[i, 1]]
            element_type = walls_array[i, 2].upper()
            member_nodes = walls_array[i, 3]
            split_nodes = member_nodes.split("-")
            final_nodes = [nodes[int(split_node)] for split_node in split_nodes]
            wall_members.append(
                WallMember(
                    num=member_num,
                    section=wall_section,
                    include_softening=include_softening,
                    element_type=element_type,
                    nodes=tuple(final_nodes)
                )
            )
    return wall_members


def create_joint_loads(example_name):
    general_info = get_general_properties(example_name)
    if not general_info.get("dynamic_analysis"):
        joint_loads_path = os.path.join(examples_dir, example_name, joint_loads_file)
        joint_loads = []
        try:
            joint_loads__array = np.loadtxt(fname=joint_loads_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
        except FileNotFoundError:
            logging.info("there is no joint load")
            return []

        for i in range(joint_loads__array.shape[0]):
            joint_loads.append(
                JointLoad(
                    node=int(joint_loads__array[i, 0]),
                    dof=int(joint_loads__array[i, 1]),
                    magnitude=joint_loads__array[i, 2],
                )
            )
        return joint_loads
    else:
        return []


def create_distributed_load(example_name):
    general_info = get_general_properties(example_name)
    if not general_info.get("dynamic_analysis"):
        distributed_load_path = os.path.join(examples_dir, example_name, distributed_load_file)
        try:
            distributed_load_magnitude = np.loadtxt(fname=distributed_load_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=float)
        except FileNotFoundError:
            logging.info("there is no distributed load")
            return None
        return DistributedLoad(magnitude=distributed_load_magnitude[0])
    else:
        return None


def create_dynamic_loads(example_name):
    general_info = get_general_properties(example_name)
    if general_info.get("dynamic_analysis"):
        dynamic_joint_loads_file = f"{dynamic_loads_dir}/joint_loads.csv"
        dynamic_loads_time_file = f"{dynamic_loads_dir}/time.csv"
        dynamic_loads_time_path = os.path.join(examples_dir, example_name, dynamic_loads_time_file)
        dynamic_joint_loads_path = os.path.join(examples_dir, example_name, dynamic_joint_loads_file)

        dynamic_joint_loads_array = np.loadtxt(fname=dynamic_joint_loads_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
        time = np.loadtxt(fname=dynamic_loads_time_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=0, dtype=float)
        dynamic_loads = []
        for i in range(dynamic_joint_loads_array.shape[0]):
            dynamic_load_dir = f"{dynamic_loads_dir}/{dynamic_joint_loads_array[i, 0]}.csv"
            dynamic_load_path = os.path.join(examples_dir, example_name, dynamic_load_dir)
            load = np.loadtxt(fname=dynamic_load_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=0, dtype=float)
            dynamic_loads.append(
                DynamicLoad(
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
    # joint_load_path = os.path.join(examples_dir, example_name, static_joint_loads_file)
    load_limit_path = os.path.join(examples_dir, example_name, load_limits_file)
    disp_limits_path = os.path.join(examples_dir, example_name, disp_limits_file)

    disp_limits = np.loadtxt(fname=disp_limits_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    load_limit = np.loadtxt(fname=load_limit_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=float)

    # joint_loads = np.loadtxt(fname=joint_load_path, usecols=range(3), delimiter=",", ndmin=2, skiprows=1, dtype=float)
    joint_loads = create_joint_loads(example_name)
    dynamic_loads = create_dynamic_loads(example_name)
    distributed_load = create_distributed_load(example_name)

    general_properties = get_general_properties(example_name)
    structure_type = general_properties["structure_type"].upper()

    include_softening = False
    if general_properties["inelastic"]["enabled"]:
        if general_properties["inelastic"]["include_softening"]:
            include_softening = True

    node_dofs_count = StructureNodeDOF[structure_type]
    initial_nodes = create_initial_nodes(example_name, structure_dim=general_properties["structure_dim"])
    nodal_boundaries = create_nodal_boundaries(example_name, initial_nodes=initial_nodes)
    linear_boundaries = create_linear_boundaries(example_name, initial_nodes=initial_nodes)

    truss2d_members = create_truss2d_members(
        example_name=example_name,
        nodes=initial_nodes,
        general_properties=general_properties,
        node_dofs_count=node_dofs_count,
        include_softening=include_softening,
    )

    frame2d_members = create_frame2d_members(
        example_name=example_name,
        nodes=initial_nodes,
        general_properties=general_properties,
        node_dofs_count=node_dofs_count,
        include_softening=include_softening,
    )

    frame3d_members = create_frame3d_members(
        example_name=example_name,
        general_properties=general_properties,
        nodes=initial_nodes,
        node_dofs_count=node_dofs_count,
        include_softening=include_softening,
    )

    plate_members = create_plate_members(
        example_name=example_name,
        nodes=initial_nodes,
        node_dofs_count=node_dofs_count,
        include_softening=include_softening,
    )

    wall_members = create_wall_members(
        example_name=example_name,
        nodes=initial_nodes,
        node_dofs_count=node_dofs_count,
        include_softening=include_softening,
    )

    limits = {
        "load_limit": load_limit,
        "disp_limits": disp_limits
    }

    loads = {
        "joint_loads": joint_loads,
        "dynamic_loads": dynamic_loads,
        "distributed_load": distributed_load,
    }
    input = {
        "structure_type": structure_type,
        "general_properties": general_properties,
        "initial_nodes": initial_nodes,
        "node_dofs_count": node_dofs_count, 
        "members": truss2d_members + frame2d_members + frame3d_members + plate_members + wall_members,
        "nodal_boundaries": nodal_boundaries,
        "linear_boundaries": linear_boundaries,
        "loads": loads,
        "limits": limits,
    }
    return input


def get_loads_input(example_name):
    dynamic_loads = create_dynamic_loads(example_name)
    joint_loads = create_joint_loads(example_name)
    distributed_load = create_distributed_load(example_name)
    input = {
        "joint_loads": joint_loads,
        "dynamic_loads": dynamic_loads,
        "distributed_load": distributed_load,
    }
    return input
