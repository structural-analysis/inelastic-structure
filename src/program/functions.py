from src.settings import settings

computational_zero = settings.computational_zero


def zero_out_small_values(array):
    low_values_flags = abs(array) < computational_zero
    array[low_values_flags] = 0
    return array
