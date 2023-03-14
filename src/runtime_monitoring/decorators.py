import functools
import os
import time


def log_output(function):
    """Saves a string output of a function to a file."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        output = function(*args)

        output_path = os.path.join("/home/ubuntu/active-workspace/koopacar-system/src/monitoring/functions/", function.__name__)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        file_path = os.path.join(output_path)  # TODO: add timestamp

        with open(file_path, 'a') as file:
            file.write(str(output))

        return output
    return wrapper()
