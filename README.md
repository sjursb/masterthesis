# Master's Thesis
Report and files related to my Master's thesis in Industrial Mathematics submitted July 4, 2022.

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

The repo also includes a licence (MIT), `.gitignore`, `requirements.txt` and `environment.yml`.
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

## Thesis Errata
- pp. 52: 
  - The starting approximations of the stage approximations should be denoted $Z_0=0$ rather than $z^0=0$
  - In approximation of the Jacobian (step two of the IRK iteration procedure), $y_n$ should be replaced with $x_n$; that is, 
  $$\nabla_x f(t_n + c_i \Delta t, x_n + z_i) \approx \nabla_x f(t_n, x_n) =: df_n.$$
  - Further, on the vector field evaluations of the stage approximations: 
  $$F(Z_i) = [f(t_n + c_1 \Delta t,\; z_0 + z_{1,i}),\; f(t_n + c_2 \Delta t,\; z_0 + z_{2,i}),\; \dots]^T.$$
## Changes Post Submission
- Added docstrings to classes and functions
- Uploaded `solver.py`, which was mistakenly left out when the repository was set up (August 1)
- Errors are noted above when found
