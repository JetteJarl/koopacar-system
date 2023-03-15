import functools
import os
import time


def log_output(function):
    """
    Decorator to saves the output of a function to a file.

    Creates one file for each second.
    All outputs of the function in that second will be written to that file.
    Each output starts with a timestamp in nanoseconds.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        output = function(*args)

        output_path = os.path.join("/home/ubuntu/active-workspace/koopacar-system/monitoring/functions/", function.__name__)
        os.makedirs(output_path, exist_ok=True)

        time_stamp_seconds = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_path, function.__name__ + time_stamp_seconds + ".log")

        with open(file_path, 'a') as file:
            # transform output to string
            if isinstance(output, (int, float, str)):
                string_output = str(output)
            elif isinstance(output, int):
                string_output = str(output)

            time_stamp_nanoseconds = str(round(time.time_ns() * 1000)) + ": "
            file.write(time_stamp_nanoseconds + string_output + "\n")

        return output
    return wrapper
