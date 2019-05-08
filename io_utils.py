import os

def get_prefix():
    if os.path.exists('/local/hopan/cube'):
        prefix = '/local/hopan/cube/'
    elif os.path.exists('/scratch/hopan/cube'):
        prefix = '/scratch/hopan/cube/'
    elif os.path.exists('/project2/risi/cube'):
        prefix = '/local/hopan/cube/'
    return prefix 
