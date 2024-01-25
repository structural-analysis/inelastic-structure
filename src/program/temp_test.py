def get_point_final_pieces(selected_pieces, violated_pieces, will_in_col_piece_num_in_structure, plastic_vars_in_basic_variables):
    unchanged_vars = [will_in_col_piece_num_in_structure] + plastic_vars_in_basic_variables
    final_pieces = selected_pieces[:]
    assign_indices = [index for index, piece in enumerate(selected_pieces) if piece not in unchanged_vars]
    assign_indices.sort(reverse=True)
    for i, assign_index in enumerate(assign_indices):
        if i < len(violated_pieces):
            final_pieces[assign_index] = violated_pieces[i]
        else:
            break
    return final_pieces


selected_pieces = [2746, 2745, 2744, 2747]
violated_pieces = [2705, 2706, 2704, 2707, 2703, 2708, 2702]
will_in_col_piece_num_in_structure = 2745
plastic_vars_in_basic_variables = [2091, 2306, 1041, 3297, 2746]

# [2704, 2745, 2706, 2705]

final_pieces = get_point_final_pieces(selected_pieces, violated_pieces, will_in_col_piece_num_in_structure, plastic_vars_in_basic_variables)
print(f"{final_pieces=}")
