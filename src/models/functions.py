
def get_elements_max_dofs_num(elements_list):
    max_dofs_num = 0
    for element in elements_list:
        if element.total_dofs_num >= max_dofs_num:
            max_dofs_num = element.total_dofs_num
    return max_dofs_num
