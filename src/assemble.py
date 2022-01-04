import numpy as np


def _transform_local_2d_matrix(element_transform, element_stiffness):
    element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
    return element_global_stiffness


def assemble_2d_frame(no_elements, no_nodes, all_transform, all_stiffness, all_elements_nodes):
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
    structure_stiffness = np.zeros((3*no_nodes, 3*no_nodes))
    for eln in range(no_elements):
        element_global_stiffness = _transform_local_2d_matrix(all_transform[eln], all_stiffness[eln])
        for i in range(6):
            for j in range(6):
                ndn = (j)//3
                p = 3*all_elements_nodes[eln, ndn+1] + j % 3
                ndnn = i//3
                q = 3*all_elements_nodes[eln, ndnn+1] + i % 3
                structure_stiffness[p, q] = structure_stiffness[p, q] + element_global_stiffness[j, i]
    return structure_stiffness
