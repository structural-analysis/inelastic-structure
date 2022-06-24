
def get_elements_max_dof_num(elements_list):
    max_dof_num = 0
    for element in elements_list:
        if element.total_dofs_num >= max_dof_num:
            max_dof_num = element.total_dofs_num
    return max_dof_num
