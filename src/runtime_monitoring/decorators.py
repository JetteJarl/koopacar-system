import functools
import os
import time
import rclpy
from std_msgs.msg import Int32
from src.runtime_monitoring.runtime_monitoring.decorators_publish_node import DecoratorPublishNode


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

        output_path = os.path.join("/home/ubuntu/active-workspace/koopacar-system/monitoring/functions/", function.__name__, "output")
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


def log_time(function):
    """
    Decorator to saves the computation time of a function to a file.

    Creates one file for each second.
    All times in that second will be written to that file.
    Each output starts with a timestamp in nanoseconds.
    Warning: If used in combination with other decorators,
             this must be the most inner decorator.
             Example: @any_other_decorator
                      @log_time
                      def any_function():
                        ...
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = time.time_ns()
        output = function(*args)
        end_time = time.time_ns()
        run_time = end_time - start_time

        output_path = os.path.join("/home/ubuntu/active-workspace/koopacar-system/monitoring/functions/", function.__name__, "time")
        os.makedirs(output_path, exist_ok=True)

        time_stamp_seconds = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_path, function.__name__ + time_stamp_seconds + ".log")

        with open(file_path, 'a') as file:
            time_stamp_nanoseconds = str(round(time.time_ns() * 1000)) + ": "
            file.write(time_stamp_nanoseconds + str(run_time) + "\n")

        return output
    return wrapper

def publish_output(function):
    """Decorator to publish the output of a function to a topic."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        output = function(*args)

        message_type = None
        if isinstance(output, int):
            message_type = Int32

        rclpy.init()
        DecoratorPublishNode(output, "/function_output", message_type)
        rclpy.shutdown()

        return output
    return wrapper
