# Paths and machine features for machine used for computations

# Number of cpus used in computations (only relevant for machines w/o GPUs or TPUs)
# import multiprocessing
computation_cores = 8  # multiprocessing.cpu_count()
# Makes computational devices from CPUs, over which code can be run in parallel
xla_flag = f'--xla_force_host_platform_device_count={computation_cores}'

# Sizes used in computations
batches = computation_cores
batch_simulations = 750


# Relative paths to directories used in project
figure_path = "../master-report/report/figures"
table_path = "./results/tables"
data_path = "./processed_data/simulations"
timing_path = "./processed_data/timings"

paths = [table_path, figure_path, data_path]
path_dict = dict(
    figure_path=figure_path,
    tables_path=table_path,
    data_path=data_path
)
