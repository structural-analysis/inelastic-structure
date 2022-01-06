import numpy as np


def _transform_loc_2d_matrix_to_glob(element_transform, element_stiffness):
    element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
    return element_global_stiffness


def assemble_2d_frame(frames, global_cords):
    """[summary]

    :param no_elements: Number of Elements
    :type no_elements: int
    :param no_nodes: Nomber of Nodes
    :type no_nodes: int
    :param all_transform: transform matrix of all elements
    :type all_transform: array
    :param all_stiffness: transform matrix of all elements
    :type all_stiffness: array
    :param all_elements_nodes: an array including end nodes of elements
    :type all_elements_nodes: array
    :return: structure stiffness after assembling stiffness of elements
    :rtype: matrix
    """
    number_of_nodes = global_cords.shape[0]
    structure_stiffness = np.zeros((3 * number_of_nodes, 3 * number_of_nodes))
    for eln in range(len(frames)):
        element_global_stiffness = _transform_loc_2d_matrix_to_glob(frames[eln].t, frames[eln].k)
        for i in range(6):
            for j in range(6):
                ndn = j // 3
                # p = 3 * all_elements_nodes[eln, ndn + 1] + j % 3
                if ndn == 0:
                    p = 3 * frames[eln].start + j % 3
                else:
                    p = 3 * frames[eln].end + j % 3

                ndnn = i // 3
                if ndnn == 0:
                    q = 3 * frames[eln].start + j % 3
                else:
                    q = 3 * frames[eln].end + i % 3
                # q = 3 * all_elements_nodes[eln, ndnn + 1] + i % 3
                structure_stiffness[p, q] = structure_stiffness[p, q] + element_global_stiffness[j, i]
    return structure_stiffness
