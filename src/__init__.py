import os, multiprocessing

print('// Initializing ML Library...\n')

cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = cores

print(f'// Number of CPU cores detected: {cores}\n')

print('// Initialization successful!')
