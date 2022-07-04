# File with functions to test performance of functions in code
import timeit

import pandas as pd

from rk_builder import *
from local_variables import timing_path, computation_cores
from problems import *
from solver import truncated_wiener
from support_functions import calc_batch_distributions, calc_confidence_intervals, calc_error

def time_hbvm_construction(s_lim=(1,10), k_max=20, q=0, number=1, verbose=True):
    """Collect hbvm construction time as function of s and k for given quadrature
    
    Parameters
    ----------
    s_lim : tuple[int]
        limits of range of s number of fundamental stages used in calculations
    k_max : int
        maximum number of total stages used in calculation, by default None
    q : int
        type of quadrature (0 for Gauss and 3 for Lobatto), by default 0
    number : int
        number of times each construction is timed, by default 1
    verbose: bool
        Whether to print progress statements during runs

    Returns
    -------
    time_dict : dict[int, dict[int, np.ma.masked_array]]
        dictionary of computation times with s as key and array of construction times as value
    """
    s_range = list(range(s_lim[0], s_lim[1] + 1))
    if k_max is None: k_max = 2*s_lim[-1] - 1
    r_ranges = [list(range(k_max - s + 1)) for s in s_range]
    time_dict = dict()
    for s, r_range in zip(s_range, r_ranges):
        timings = dict(zip(range(k_max+1), np.full_like(range(k_max+1), np.nan, dtype=np.float64)))
        for r in r_range:
            k = s + r
            if verbose: print(f"Timing construction of HBVM({k}, {s})...")
            timings[(s-1)+r] = timeit.timeit(
                'RK(k=k, s=s, quadrature=q, numeric=True)', 
                number=number, 
                globals=dict(RK=RK, k=k, s=s, q=q)
            )/number # only interested in mean of iterations
    
        time_dict[f"s={s}"] = timings
    
    return time_dict


def prepare_solvers(batches=1, batch_simulations=1, sigma=1):
    """Creates dict of solvers with identical problem configurations."""

    # Prepare features with identical settings
    henon_heiles.batches = henon_heiles_old.batches = henon_heiles_autodiff.batches = batches
    henon_heiles.batch_simulations = henon_heiles.batch_simulations = henon_heiles.batch_simulations = batch_simulations
    henon_heiles.sigma = henon_heiles_old.sigma = henon_heiles_autodiff.sigma = sigma

    # Prepare dict of solvers
    solver_dict = dict(
        new=henon_heiles.sde_solver,
        autodiff=henon_heiles_autodiff.sde_solver, # The one using atodifferentiation - pure jax
        old=henon_heiles_old.sde_solver_old, # The first implementation - pure numpy + scipy
        mixed=henon_heiles.sde_solver_old, # Old implementation with jitted vector field functions - mix of numpy/scipy and jax 
    )

    return solver_dict


def solver_timing(solver:Callable, rk:RK, t0:np.number, T:np.number, dt:float, batches:int, batch_simulations:int, number:int, sigma:float, verbose:bool=True):
    """Perform a single timing of a solver with given parameters."""
    dW = truncated_wiener(t0, T, dt, batches=batches, batch_simulations=batch_simulations, k=rk.k)
    var_dict = dict(solver=solver, rk=rk, dt=dt, dW=dW, t0=t0, T=T, sigma=sigma)
    
    timing = timeit.timeit("solver(rk, dt, dW=dW, t0=t0, T=T, sigma=sigma)", number=number, globals=var_dict)/number
    if verbose: print(f"Solver clocked in at an average {timing} seconds.")
    return timing

def time_timesteps(rk=RK(s=3, k=6), power_lim = (1,6), batches=1, batch_simulations=1, sigma=1., number=1, verbose=True):
    """Collect solver time for old and new implementation as function of number of timesteps """
    dt = 1e-2
    timesteps_list = [10**i for i in range(power_lim[0], power_lim[1] + 1)]
    t0 = 0
    Ts = np.array(timesteps_list) * dt

    solver_dict = prepare_solvers(batches=batches, sigma=sigma) # Using batches=computation_cores might give bad comparison
    time_dict = dict()

    for solver_name, solver in solver_dict.items():
        timings = dict(zip(timesteps_list, np.zeros_like(timesteps_list)))

        for T, timesteps in zip(Ts, timesteps_list):
            if verbose: print(f"\nTiming calculations using {solver_name} solver over {timesteps} timesteps...")            
            timings[timesteps] = solver_timing(solver, rk, t0, T, dt, batches, batch_simulations, number, sigma, verbose)
        
        time_dict[solver_name] = timings
    
    return time_dict

def time_simulations(rk=RK(s=3, k=6), power_lim=(0,5), timesteps=10, batches=1, sigma=1., number=1, verbose=True):
    """Collect solver time for old and new implementation as function of number of simulations"""
    t0 = 0.; T = 1.; dt = (T-t0)/timesteps

    simulations_list = [10**i for i in range(power_lim[0], power_lim[1]+1)]
    solver_dict = prepare_solvers(batches=batches, sigma=sigma)
    time_dict = dict()

    for solver_name, solver in solver_dict.items(): 
        timings = dict(zip(simulations_list, np.zeros_like(simulations_list, dtype=np.float64)))

        for batch_simulations in simulations_list:
            if verbose: print(f"\nTiming calculations using {solver_name} solver over {batch_simulations} simulations...")
            timings[batch_simulations] = solver_timing(solver, rk, t0, T, dt, batches, batch_simulations, number, sigma, verbose=verbose)
        
        time_dict[solver_name] = timings

    return time_dict

def time_silent_stages(s=4, r_max=10, q=0, timesteps=100, batches=1, batch_simulations=100, sigma=1., number=1, verbose=True):
    t0 = 0.; T = 1; dt = (T - t0) / timesteps

    r_list = [i for i in range(r_max+1)]
    rk_list = [RK(k=s+r, s=s, quadrature=q) for r in r_list]
    solver_dict = prepare_solvers(batches=batches, sigma=sigma)
    time_dict = dict()

    for solver_name, solver in solver_dict.items():
        timings = dict(zip(r_list, np.zeros_like(r_list, dtype=np.float64)))

        for r, rk in zip(r_list, rk_list):
            if verbose: print(f"\nTiming calculations using {solver_name} solver with {rk.method_name}) ...")
            timings[r] = solver_timing(solver, rk, t0, T, dt, batches, batch_simulations, number, sigma, verbose=verbose)
        
        time_dict[solver_name] = timings

    return time_dict


def time_batches(rk=RK(s=3, k=6), batches_max=computation_cores, timesteps=100, batch_simulations=100, sigma=1., number=1, verbose=True):
    """Collect solver time for old and new implementation as function of number of batches"""
    t0 = 0.; T = 1.; dt = (T-t0)/timesteps

    batches_list = [i for i in range(1, batches_max+1)]

    solver_dict = prepare_solvers(batches=computation_cores, sigma=sigma)
    time_dict = {solver_name:dict() for solver_name in solver_dict.keys()}

    for batches in batches_list: # NB! Opposite loop order!
        for solver_name, solver in solver_dict.items():
            if verbose: print(f"\nTiming calculations using {solver_name} solver over {batches} batches...")
            timing = solver_timing(solver, rk, t0, T, dt, batches, batch_simulations, number, sigma, verbose=verbose)
            time_dict[solver_name][batches] = timing

    return time_dict 

def timings(time_function:callable, title:str, plot_dict=dict(), number=1, timing_path="", show=True, verbose=True):
    """
    timings makes and optionally shows and/or stores plot and table of timings generated from time_function

    Plot is created with given title, tight layout and and ylabel 'time (seconds)' as only set parameters; the rest must be supplied in plot_dict.

    Parameters
    ----------
    time_function : callable
        function timing execution of some routine taking number and verbose as input, returning timings as a nested dictionary
    title : str
        title used for plot and table (and save name)
    plot_dict : dict[str, Any], optional
        extra plot setting, by default dict()
    number : int, optional
        number of iterations for which the execution is timed, by default 1
    timing_path : str, optional
        relative path to folder where the timings are stored, by default ""
    show : bool, optional
        whether to show result at the end after storing, by default True
    verbose : bool, optional
        whether to print helpful statments, by default True
    """
    time_dict:"dict[str, dict[int, np.ndarray]]" = time_function(number=number, verbose=verbose)
    
    # Timings dataframe
    df = pd.DataFrame(time_dict)

    # Timings figure
    fig, ax = plt.subplots(figsize=(4,3), layout="constrained")
    plot_dict = {"title":f"{title}, {number} Runs", "ylabel":"time (seconds)", **plot_dict}
    ax.set(**plot_dict)
    for label, d in time_dict.items(): ax.plot(d.keys(), d.values(), label=label, marker="x")
    ax.legend()
    
    if len(timing_path): # Saving if given path for saving figures
        save_name = f"{timing_path}/{title}"
        df.to_csv(f"{save_name}.csv")
        plt.savefig(f"{save_name}.pdf")
        if verbose: print(f"Saved results at {save_name} as .csv and .pdf")
    
    if show:
        print(df)
        plt.show()

def prepare_problems(**kwargs):
    problem_dict = {
        "Hénon-Heiles":dict(autodiff=henon_heiles_autodiff, expression=henon_heiles),
        "Double-Well":dict(autodiff=double_well_autodiff, expression=double_well),
        #"Kepler":dict(autodiff=kepler_2d_autodiff, expression=kepler_2d),
        "H order six":dict(autodiff=h6_autodiff, expression=h6),
    }

    for problem_set in problem_dict.values():
        for problem in problem_set.values():
            problem.adjust_parameters(**kwargs)

    return problem_dict

def autodiff_precision(rk=RK(k=4, s=2, quadrature=0), dt=1e-2, t0=0, T=100, sigma=1.0, alpha=0.1, batches=4, batch_simulations=1000, verbose=True):
    """Make figure show that the path difference between autodiff and new solver for a set of problems is close to zero"""

    problem_dict = prepare_problems(batches=batches, batch_simulations=batch_simulations, T=T, t0=t0, sigma=sigma)
    # Possibly use different seeds for differen problems? might not be that important
    
    diff_dict = dict()
    i = 0
    for problem_name, problem_set in problem_dict.items():
        print(f"Calculating solutions for {problem_name}...")
        dW = truncated_wiener(t0=t0, T=T, dt=dt, batches=batches, batch_simulations=batch_simulations, seed=100+i, k=rk.s, truncate=True)
        timesteps = dW.shape[0]
        x_ref = problem_set['expression'].sde_solver(rk, dt, dW=dW)
        x_autodiff = problem_set['autodiff'].sde_solver(rk, dt, dW=dW)
        difference = calc_error(x_autodiff, x_ref, g=lambda x: 0, strong=True, global_error=None).reshape((timesteps+1, batches * batch_simulations))
        diff_dict[problem_name] = calc_confidence_intervals(difference, alpha=alpha)
        i +=1
    t = np.linspace(t0, T, timesteps+1)
    color_dict = {"Hénon-Heiles":"red", "Double-Well":"blue", "Kepler":"green", "H order six":"purple"}
    fig, ax = plt.subplots()
    ax.set(title=f"Autodiff Comparison, {(1-alpha)*100}% Confidence", xlabel="t", ylabel=R"$\sqrt{{E\left[||x_{{autodiff}} - x_{{expression}}||^2\right]}}$", yscale='linear')
    for problem_name, diff_set in diff_dict.items():
        diff, diff_deviation, _ = diff_set
        ax.plot(t, diff, label=problem_name, color=color_dict[problem_name])
        ax.fill_between(t, diff+diff_deviation, diff-diff_deviation, color=color_dict[problem_name], alpha=0.3, linewidth=0, zorder=1.5)

    ax.legend()
    
    plt.show()


def csv_to_latex(filename, path):
    full_path = f"{path}/{filename}"
    df = pd.read_csv(full_path, header=0, index_col=0)
    df.to_latex(full_path, column_format="l|cccc", index=True, float_format="%.3f")
    print(f"Successfully converged '{full_path}' to .tex")




if __name__ == "__main__":
    number = 10

    # RK construction timings #TODO: run again to double-check results.
    # q_dict = {0:"Gauss", 3:"Lobatto"}
    # for quadrature in [0,3]:
    #     timings(
    #         lambda number, verbose: time_hbvm_construction(s_lim=(1,10), k_max=20, q=quadrature, number=number, verbose=verbose), 
    #         title=f"Butcher Table Construction Time for {q_dict[quadrature]}-HBVMs", 
    #         plot_dict=dict(xlabel=r"total stages $k$"), number=number
    #     )

    
    # timesteps timings
    # timings(time_timesteps, title="Variable Timesteps Timings", plot_dict=dict(xscale="log", yscale="log", xlabel="timesteps"), timing_path=timing_path, number=number, show=False)

    # simulation timings
    # timings(time_simulations, title="Variable Simulations Timings", plot_dict=dict(xlabel="simulations", xscale="log", yscale="log"), timing_path=timing_path, number=number, show=False)

    # silent stages timings
    # timings(time_silent_stages, title="Variable Stages Timings", plot_dict=dict(xlabel="silent stages"), number=number, timing_path=timing_path, show=False)

    # batches timings
    # timings(time_batches, title="Variable Batches Timings", plot_dict=dict(xlabel="batches"), number=number, timing_path=timing_path, show=True)

    csv_to_latex("Variable Stages Timings.csv", timing_path)
    # autodiff_precision()
