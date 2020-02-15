import sys
import logging

def get_logger(fname=None, stdout=True, tofile=True):
    '''
    fname: file location to store the log file
    '''
    handlers = []
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)
    if tofile:
        file_handler = logging.FileHandler(filename=fname)
        handlers.append(file_handler)

    str_fmt = '[%(asctime)s.%(msecs)03d] %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.DEBUG,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger
