import re


def list_from_file(file_path):
    """Converts [x, y, z] points from file into list"""
    try:
        file = open(file_path, "r")

        data = []

        for line in re.split("\n", file.read()):
            if line == '':
                continue

            data.append(list(map(float, re.split(",", line[1:-1]))))

        return data

    except OSError:
        print("Can not open/read the file: " + file_path)