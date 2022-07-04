# Paths and computational cores and such

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

computation_cores = 8
xla_flag = f'--xla_force_host_platform_device_count={computation_cores}' # max 8 on my laptop, max 28 on markov (weird number)

batches = computation_cores
batch_simulations = 750
