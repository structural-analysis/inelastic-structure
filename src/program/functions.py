from src.settings import settings


def zero_out_small_values(array):
    low_values_flags = abs(array) < settings.computational_zero
    array[low_values_flags] = 0
    return array
