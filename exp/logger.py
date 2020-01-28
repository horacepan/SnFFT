import logging

def get_logger(fname):
    '''
    fname: file location to store the log file
    '''
    str_fmt = '[%(asctime)s.%(msecs)03d] %(levelname)s %(module)s: %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        filename=fname,
        level=logging.DEBUG,
        format=str_fmt,
        datefmt=date_fmt)

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(str_fmt, datefmt=date_fmt)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
