import os

def get_prefix():
    if os.path.exists('/local/hopan/'):
        prefix = '/local/hopan/'
    elif os.path.exists('/scratch/hopan/'):
        prefix = '/scratch/hopan/'
    elif os.path.exists('/project2/risi/'):
        prefix = '/project2/risi/'
    return prefix 
