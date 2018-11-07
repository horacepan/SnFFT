import os
import psutil
import resource

def check_memory():
    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    process = psutil.Process(os.getpid())
    resp = process.memory_info().rss
    print("Consumed {}mb memory | {:.2f}kb".format(res_/(1024**2), res_/(1024)))
