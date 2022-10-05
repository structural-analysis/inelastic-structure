
def get_members_max_dofs_num(members_list):
    max_dofs_num = 0
    for member in members_list:
        if member.total_dofs_num >= max_dofs_num:
            max_dofs_num = member.total_dofs_num
    return max_dofs_num
