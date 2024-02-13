import numpy as np


def normalize_real_data(arrays):
    arrays = list(map(np.array, arrays))
    number_of_points = min(map(len, arrays))
    groups_for_arrays = [np.array_split(array, number_of_points) for array in arrays]
    values = [[np.mean(group) for group in groups] for groups in groups_for_arrays]
    values = list(map(np.array, values))
    return values


def find_region_of_main_peak(values_array):
    i_beg_min = None
    i_end_max = None
    for values, _ in values_array:
        i_max = np.argmax(values)
        i_beg = i_max
        i_end = i_max
        while i_beg > 0:
            i_beg -= 1
            if values[i_beg] < 0:
                break
            vii = (values[i_beg - 1] + values[i_beg + 1]) / values[i_beg] - 2
            if vii >= 1e-6:
                break
            if values[i_beg - 1] > values[i_beg]:
                break
        if i_beg_min is None or i_beg < i_beg_min:
            i_beg_min = i_beg
        while i_end < len(values) - 1:
            i_end += 1
            if values[i_end] < 0:
                break
            vii = (values[i_end - 1] + values[i_end + 1]) / values[i_end] - 2
            if vii >= 1e-6:
                break
            if values[i_end - 1] > values[i_end]:
                break
        if i_end_max is None or i_end > i_end_max:
            i_end_max = i_end
    return i_beg_min, i_end_max
