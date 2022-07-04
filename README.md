# Master's Thesis
Report and files related to my Master's thesis in Industrial Mathematics submitted June 4, 2022.

## Repo Summary
Folder tree:
```sh
masterthesis
├── processed_data
│   ├── butcher_tables
│   └── convergence_data
├── reports
├── results
│   ├── butcher_tables
│   ├── convergence_tables
│   ├── figures
│   └── timings
└── src
    └── testing
        └── __pycache__
```

The repo also includes a licence (MIT), `.gitignore`, `requirements.txt`
### processed_data
- some raw data on Butcher tables
- convergence data, which was computed on an external server

### reports
Contains three files: 
- `specialization_project.pdf`, the report written before the Master's thesis
- `thesis_final.pdf`, which was submitted for assessment
- `thesis_final_b5.pdf`, the above converted to  B5 format and with three minor corrections

### results
Contains all figures and tables used in the Master's thesis.

### src
All the code files related to the project, as well as an (incompleted) test suite.
The code is slightly cluttered and poorly documented at the time of submission, unfortunately.

## Setting up code
The `jax` package does not run on Windows. If you use a Windows computer, the environment has to be installed on a Linux subsystem (WSL).

