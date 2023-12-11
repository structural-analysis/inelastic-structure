from src.settings import settings


def zero_out_small_values(array):
    low_values_flags = abs(array) < settings.computational_zero
    array[low_values_flags] = 0
    return array


def print_specific_properties(obj_list, properties):
    # Usage
    # properties_to_print = ['num_in_yield_point']
    # print_specific_properties(list_of_objects, properties_to_print)

    for obj in obj_list:
        values = [getattr(obj, prop) for prop in properties]
        print(*values)
