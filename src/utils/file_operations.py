import re

import numpy as np


def list_from_file(file_path):
    """Converts [x, y, z] points from file into list"""
    array_regex = re.compile("\[.*?]")

    file = open(file_path, 'r')
    content = file.read().replace('\n', '')

    array_strings = array_regex.findall(content)
    data = np.array([np.fromstring(arr_str[1:-1], sep=',') for arr_str in array_strings])

    file.close()

    return data


