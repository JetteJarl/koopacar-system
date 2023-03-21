import time
import pickle

from src.runtime_monitoring.decorators import *
from src.runtime_monitoring.runtime_monitoring.central_monitoring_node import CentralMonitoringNode


@CentralMonitoringNode.save_function_output
def power(base, exponent):
    return base ** exponent




#for i in range(10000):
    #power(2, i)

print(power(2, 3))

#time_stamp = time.time_ns()
#time_stamp = time.localtime()
#print(time_stamp)

