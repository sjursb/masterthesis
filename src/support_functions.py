# Support functions for methods in problem_classes.py
from distutils.command.build import build
import os
from time import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats

from local_variables import xla_flag
# To use cpu cores in calculations:
os.environ['XLA_FLAGS'] = xla_flag # sets number of virtual logical devices

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit

from rk_builder import RK
from solver import irk_sde_solver, truncated_wiener, irk_ode_solver

class HumanBytes:
    """Class copied from stackoverflow to get human-readable byte format
    
    Link: https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb"""
    METRIC_LABELS: "list[str]" = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    BINARY_LABELS: "list[str]" = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    PRECISION_OFFSETS: "list[float]" = [0.5, 0.05, 0.005, 0.0005] # PREDEFINED FOR SPEED.
    PRECISION_FORMATS: "list[str]" = ["{}{:.0f} {}", "{}{:.1f} {}", "{}{:.2f} {}", "{}{:.3f} {}"] # PREDEFINED FOR SPEED.

    @staticmethod
    def format(num:"int|float", metric: bool=False, precision: int=2) -> str:
        """
        Human-readable formatting of bytes, using binary (powers of 1024)
        or metric (powers of 1000) representation.
        """

        assert isinstance(num, (int, float)), "num must be an int or float"
        assert isinstance(metric, bool), "metric must be a bool"
        assert isinstance(precision, int) and precision >= 0 and precision <= 3, "precision must be an int (range 0-3)"

        unit_labels = HumanBytes.METRIC_LABELS if metric else HumanBytes.BINARY_LABELS
        last_label = unit_labels[-1]
        unit_step = 1000 if metric else 1024
        unit_step_thresh = unit_step - HumanBytes.PRECISION_OFFSETS[precision]

        is_negative = num < 0
        if is_negative: # Faster than ternary assignment or always running abs().
            num = abs(num)

        for unit in unit_labels:
            if num < unit_step_thresh:
                # VERY IMPORTANT:
                # Only accepts the CURRENT unit if we're BELOW the threshold where
                # float rounding behavior would place us into the NEXT unit: F.ex.
                # when rounding a float to 1 decimal, any number ">= 1023.95" will
                # be rounded to "1024.0". Obviously we don't want ugly output such
                # as "1024.0 KiB", since the proper term for that is "1.0 MiB".
                break
            if unit != last_label:
                # We only shrink the number if we HAVEN'T reached the last unit.
                # NOTE: These looped divisions accumulate floating point rounding
                # errors, but each new division pushes the rounding errors further
                # and further down in the decimals, so it doesn't matter at all.
                num /= unit_step

        return HumanBytes.PRECISION_FORMATS[precision].format("-" if is_negative else "", num, unit)


### BaseODE-related functions

def calc_silent_max(nu, s, silent_max):
    """Calculate number of silent stages HBVM of degree 2*s to preserve polynomian Hamiltonian of order nu exactly.
    
    Notes:
     - Returns minimum value of calculated number of silent stages r_max and supplied silent_max
     - non-polynomial H (i.e. nu=None) returns silent_max"""
    if nu is None: nu = np.infty
    r_max = s * (np.ceil(nu/2) - 1)
    return int(min(r_max, silent_max))

def calc_exact_solution(exact, t0, T, timesteps, sigma_0, sigma=None, dW=None):
    t = jnp.linspace(t0, T, timesteps+1)
    if dW is not None:
        W = np.zeros((timesteps+1, *dW.shape[1:]))
        W[1:] = jnp.cumsum(dW, axis=0)
    else: W = dW
    reference = exact(t, W, sigma_0=sigma_0, sigma=sigma)
    return np.array(reference)


def calc_ode(self, hbvms:"list[RK]", base_max_ref:"tuple[np.number, np.number, np.number]"=None):
    """calc__ode calculates approximated solutions using supplied hbvms, and stepsize based on base_max_ref.

    Parameters
    ----------
    hbvms : list[RK_butcher], optional
        Instances of RK_butcher used in approximation, by default
    base_max_ref: tuple
        tuple of (base, power_max, power_ref) used to set up dt used in calculation and reference (power_max and power_ref, respectively)

    Returns
    -------
    xs: list[np.ndarray]
        approximations corresponding to the methods in hbvms
    x_exact: np.ndarray
        "exact solution of length timesteps + 1 (same as each x in xs) 
    """
    
    # Time features
    if base_max_ref is None:
        base, power_max, power_ref = self.base, self.power_max, self.power_ref
    else:
        base, power_max, power_ref = base_max_ref
    dt = 1 / self.base ** power_max
    t0, T = self.t0, self.T
    timesteps = int((T-t0)/dt)

    # Calculating solutions
    xs = [irk_ode_solver(self.x0, hbvm, self.f, self.df, t0=t0, T=T, dt=dt) for hbvm in hbvms]

    # Calculate "exact" solution
    if self.x_ref is None:
        self.generate_reference((base, power_ref))
    step = base ** (power_ref - power_max)
    x_exact = self.x_ref[::step]

    return xs, x_exact
        

def calc_hamiltonian(self, xs, x_exact, use_initial_value=True):
    """Calculates Hamiltonian function of ODE approximations and exact hamiltonian using initial value H(x0) or x_exact."""
    t = jnp.linspace(self.t0, self.T, len(x_exact))
    # Calculate Hamiltonian
    Hx = [self.hamiltonian(x) for x in xs]
    if use_initial_value:
        H_exact = self.hamiltonian(jnp.aray(self.x0)) * jnp.ones_like(x_exact[..., 0])
    else:
        H_exact = self.hamiltonian(x_exact)

    return Hx, H_exact


def calc_error(x, x_exact:np.ndarray, g:callable, strong:bool, global_error:bool, use_initial_value=False, x0=None):
    """Calculate specified type of error (strong/weak and global/local) for input approximation x and exact solution x_exact."""
    if strong:          error = jnp.linalg.norm(x_exact - x, ord=2, axis=-1) # Strong error
    else:
        if use_initial_value: 
            if x0 is None: 
                d = x_exact.shape[-1]
                x0 = x_exact.flat[:d]
            gx_exact = np.full(x_exact.shape[:-1], g(x0))
        else: gx_exact = g(x_exact)
        error = gx_exact-g(x) # Usually Hamiltonian error

    if global_error is None:    return error
    elif global_error:          return error[-1] # np.sum(error) gives wrong result, as it is the sum of global errors after n timesteps!!
    else:                       return error[1] # np.mean(error) is the mean of global errors after n timesteps, whereas error[0] always returns 0!!

def calc_convergence(self, hbvms:"list[RK]", stochastic=False, verbose=False, dW_ref=None, delete_dW_ref=False, use_initial_value=True, path=""):
    """Calculates errors for given HBVMs using different stepsize for use in convergence plot.
    
    Works both for ODEs (stochastic=False) and SDEs (stochastic=True)."""
    dt_ref, dts = self.get_dts()
    K, L = len(hbvms), len(dts)
    error_shape = [2, 2, K, L]
    if not self.sigma: stochastic = False
    if self.x_ref is None: self.generate_reference(verbose=verbose, dW_ref=dW_ref, stochastic=stochastic)

    if stochastic:

        step = int(self.base ** (self.power_ref - self.power_max))
        shape_temp = (self.dW.shape[0]//step, step, *self.dW.shape[1:])
        dW = jnp.sum(jnp.reshape(self.dW, shape_temp), axis=1)
        batches, batch_simulations = shape_temp[1:3]
        dW_list = self.get_dW_list(self.dW)
        if delete_dW_ref:
            dW_ref = None 
            self.dW = None
        solver = lambda hbvm, dt, dW: self.sde_solver(hbvm, dt, dW=dW, verbose=verbose) # TODO: Define for AdditiveDrift
        error_shape +=[self.batches]
        error_deviations = np.zeros(error_shape, dtype=object) # inside if as it isn't relevant to odes

    else:
        solver = lambda hbvm, dt, dW: self.ode_solver(hbvm, dt) # To get same syntax as for sde case
        dW, dW_ref = None, None # unused
    
    errors = np.zeros(error_shape) # Each object a list of mean of errors in batches

    for k in range(K):
        if verbose: print("Calculating errors of {}:".format(hbvms[k].method_name))
        time0 = time()
        for l in range(L):
            if stochastic: dW = dW_list[l]
            # Calculate solution
            if verbose: print(f"Using stepsize {dts[l]}...")
            x = jnp.array(solver(hbvms[k], dts[l], dW))
            

            step = int(dts[l]/dt_ref)
            x_exact = jnp.array(self.x_ref[::step]).reshape(x.shape)

            if self.hamiltonian is None: g = lambda x: jnp.linalg.norm(x, ord=2, axis=-1)
            else: g = self.hamiltonian

            for strong in range(2):
                for global_error in range(2):
                    
                    error = calc_error(x, x_exact, g, strong=strong, global_error=global_error, use_initial_value=use_initial_value, x0=self.x0)
                    if stochastic: # Calculate mean of and confidence intervals for mean errors in batches
                        error, error_deviation = calc_batch_distributions(error) # mean and std in each of the batches
                        error_deviations[strong, global_error, k, l] = error_deviation
                    
                    # TODO: move the below routine into calc_error in some way (or into calc_batch_distributions, alternatively)
                    if strong: error = np.sqrt(error) # To correct for squares in calc_error
                    else: error = np.abs(error) # To ensure positive weak error
                    errors[strong, global_error, k, l] = jnp.abs(error) # jnp.abs here ensures weak error behaves as expected
        if verbose:print(f"Finished calculating error convergence of {hbvms[k].method_name} in {time()-time0} seconds.\n")
        
    if not stochastic: error_deviations = np.full(errors.shape[:3], None)
    if path and stochastic: self.store_computed_errors(hbvms, np.array(dts), errors, error_deviations, path=path) # Only interesting to store for stochastic calculations, as these take more than 10 seconds to compute
    return dts, errors, error_deviations



def calc_ode_k_convergence(self, base_max_ref:"tuple[int, int, int]"=None, s_min_max:"tuple[int, int]"=None, use_initial_value=False, global_error=True):
    # FIXME: Old code, needs to be updated to work with current repo!
    if base_max_ref is None: base_max_ref = (self.base, self.power_max, self.power_ref)
    if s_min_max is None: s_range = self.s_range
    else:
        s_min, s_max = s_min_max
        s_range = range(s_min, s_max + 1)
        if s_max >= self.rk_ref.s: self.rk_ref = RK(s=s_max + 1)
        self.generate_reference(base_max_ref[::2], verbose=True)
    silent_stages = range(self.silent_max+1) # np.arange doesn't work with param.Integer
    
    Q = len(self.quadratures)
    errors = []
    labels = []
    quad_dict = {0:"Gauss", 3:"Lobatto"}
    for s in s_range:
        for i in range(Q):
            hbvms = [RK(k=s+r, s=s, quadrature=self.quadratures[i]) for r in silent_stages]
            xs, x_exact = self.calc_solutions(hbvms, base_max_ref)
            Hx, H_ref = self.calc_hamiltonian(xs, x_exact, use_initial_value=use_initial_value)
            error = np.zeros(self.silent_max + 1)

            for r in silent_stages:
                error_temp = np.abs(Hx[r] - H_ref)
                if global_error: error[r] = np.sum(error_temp)
                else: error[r] = np.mean(error_temp)
    
            errors.append(error)
            labels.append("{}-HBVM(k, {})".format(quad_dict[self.quadratures[i]], s))

    
    return np.array(errors), labels

def calc_relative_convergence(hbvms:"list[RK]", e:np.ndarray, hbvm_refs:"list[RK]", e_ref:np.ndarray):
    """Compare Strong/Global ODE error for HBVMs with reference collocation method of same order (i.e. Gauss-2s)."""

    strong, global_error = 1, 1 # only interested in error in ODE/SDE, not Hamiltonian, and global behaviour (more accurate).
    errors = e[strong, global_error]
    error_refs = e_ref[strong, global_error]
    ref_dict = {hbvm.s:error_ref for hbvm, error_ref in zip(hbvm_refs, error_refs)}

    # Calculate relative errors times parameter s to separate methods of different order in same plot
    relative_error_list = [hbvm_error / ref_dict[hbvm.s] for hbvm, hbvm_error in zip(hbvms, errors)]
    
    return np.array(relative_error_list)

            
def calc_convergence_order(errors:np.ndarray, base:np.number=2., order_tol=1e-14, decimals=2, increasing=True):
    """calc_observed_order calculates observed convergence order of errors for values larger than tol

    Assumes `error = O( base ** (-c * i)), i=1,2,...`

    with stepsize `dt = 1 / base ** (i)`.


    Parameters
    ----------
    errors : 
        list of errors calculated with decreasing stepsize
    base : np.number, optional
        base of the stepsize, by default 2.
    tol : float, optional
        lowest value of errors used in order calculation, by default 1e-15
    decimals : int, optional
        number of decimals in returned order calculation, by default 2
    increasing : bool, optional
        whether the errors are reported in order of increasing stepzise, by default True

    Returns
    -------
    float
        calculated error convergence
    """
    e = errors[errors > order_tol]
    if increasing: e = e[-1::-1] # in case the errors are offered for increasing stepsize (which is default for convergence functions)
    if len(e) < 2: order =  np.nan # Not enough values to calculate...
    else:
        e1, e2 = e[:-1], e[1:]
        order:float = np.round(np.mean(np.log(e1 / e2)) / np.log(base), decimals=decimals)
    return order

def calc_expected_order(hbvm:RK, stochastic, global_error, strong, nu=None, silent_max=100):
    k_conserved = hbvm.s + calc_silent_max(nu, hbvm.s, silent_max)
    conserved = hbvm.k >= k_conserved
    if strong: order:int = hbvm.s
    else: 
        if not conserved: order = hbvm.k
        else: order = np.nan
    if not stochastic: order *= 2
    if not global_error: order += 1
    return order

def get_order_table(hbvms:"list[RK]", errs, stochastic, global_error, nu=None, base=2., filename="order_table", order_tol=1e-14, store_table_path=""):

    # Set up table data
    method_names = [hbvm.method_name for hbvm in hbvms]
    data = np.zeros((len(hbvms), 4))

    for k in range(len(hbvms)):
        for strong in range(2):
            for expected in range(2):
                if expected: order = calc_expected_order(hbvms[k], stochastic, global_error, strong=strong, nu=nu)
                else: order = calc_convergence_order(errs[strong, k], base=base, order_tol=order_tol)

                j = int(strong)*2 + int(not expected) # hamiltonian and expected first
                data[k, j] = order
    
    data = np.ma.masked_invalid(data)
    # Set up column and row names
    columns = pd.MultiIndex.from_product([["Hamiltonian", "Solution"], ["Expected", "Observed"]])
    index = pd.Index(method_names, name="Method Name")
    
    order_table = pd.DataFrame(data, index=index, columns=columns)

    if len(store_table_path):
        table_filename = f"{store_table_path}/{filename}.tex".replace("%", "")
        order_table.reset_index(inplace=False).to_latex(table_filename, index=False, na_rep='--', multicolumn_format='c', column_format='l|cccc')
        print(f"Stored order table to '{table_filename}'.")
    return order_table

### SingleIntegrand - related functions

def get_dW_list(
        dW_ref:"None|np.ndarray"=None,
        t0=0, T=1, base:int=2., power_max:int=8, comparisons:int=5,
        batch_simulations=10**2, batches=10, seed=100, k=2, diffusion_terms=0, truncate=True
    ) -> "list[np.ndarray]":
    """
    Get list of simulated Wiener increments built from a minimum stepsize acquired from a base (e.g.2) and a max_power(e.g.11),
    where each additional list is a sum over base number of increments in the former.

    As of 25.05 unused due to other preferable ways to do computations.
    """
    dt = 1 / base ** power_max
    timesteps_max = int((T-t0)/dt)
    if dW_ref is None:
        dW = truncated_wiener(t0, T, dt, batches, batch_simulations, seed, diffusion_terms, k, truncate=truncate)
        dW_shape = list(dW.shape)
    else:
        timesteps_ref, batches, batch_simulations = dW_ref.shape[:3]

        dW_shape = [timesteps_ref, batches, batch_simulations]
        if len(dW_ref.shape) == 4: dW_shape.append(dW_ref.shape[-1]) # Add diffusion_terms if there are any
        dW = np.zeros(dW_shape)

        step = int(timesteps_ref * dt)

        for j in range(timesteps_max):
            dW[j] = np.sum(dW_ref[j*step: (j+1)*step], axis=0)
    
    dW_list = []
    timesteps = (timesteps_max / base ** np.arange(comparisons)).astype(int)

    for i in range(comparisons):
        step = int(timesteps_max/timesteps[i])
        dW_shape[0] = timesteps[i]
        dW_temp = np.zeros(dW_shape)

        for j in range(timesteps[i]):
            dW_temp[j,...] = np.sum(dW[j*step:(j+1)*step, ...], axis=0)
        
        dW_list.append(dW_temp)

    return dW_list

def get_reference_list(self, reference_solution:np.ndarray, base_max_ref:"tuple[int, int, int]"):
    # Pick out only the relevant steps in time for each reference solution
    if base_max_ref is None: base, power_max, power_ref = self.base, self.power_max, self.power_ref
    else: base, power_max, power_ref = base_max_ref
    reference_list = []
    power_max = min(power_max, power_ref) # For safety
    comparisons = min(power_max, self.comparisons) # More safety
    for i in range(comparisons):
        step = int(base ** (power_ref - power_max + i))
        reference_list.append(jnp.array(reference_solution[::step]))

    return reference_list

def calc_sde(self, hbvms:"list[RK]", base_max_ref:"tuple[int]"=None, dW:"np.ndarray"=None, verbose=False)->"tuple[list[np.ndarray], np.ndarray]":
    """Analogously to calc_ode, it calculates approximations to Single-Integrand SDEs using irk_sde_solver with supplied hbvms.
    NOTE: calculated x's have shape (timesteps+1, batches, batch_simulations, d) instead of (timesteps+1, d)."""
    if not base_max_ref: base, power_max, power_ref = self.base, self.power_max, self.power_ref
    else: base, power_max, power_ref = base_max_ref
    
    dt = 1 / base ** power_max
    step = base ** (power_ref - power_max)
    # Set up simulated Wiener increments
    if dW is not None:
        dW_shape = dW.shape
    else:
        # Find dW from dW_ref
        if self.dW_ref is None: self.generate_dW(base_power_ref=(base, power_ref), verbose=verbose)
        timesteps = int((self.T-self.t0)/dt)
        step = int(self.dW_ref.shape[0]/timesteps)
        dW_shape = list(self.dW_ref.shape)
        dW_shape[0] = timesteps
        dW = np.zeros(shape=dW_shape)
        for i in range(timesteps): dW[i] = np.sum(self.dW_ref[i*step:(i+1)*step], axis=0)
    
    if verbose:
        print("Beginning to calculate approximations with step-size", dt)
        time0 = time()
    xs = [self.sde_solver(hbvm, dt, dW, verbose=verbose) for hbvm in hbvms]
    if verbose:
        print("Calculating approximations finished in {} seconds.".format(time()-time0))
    # Calculate exact solution for given timesteps
    if self.x_ref is None: self.generate_reference((base, power_ref), verbose=verbose)
    x_exact = self.x_ref[::step]
    
    return xs, x_exact


def calc_batch_distributions(errors:np.ndarray, strong=False):
    """Calculates mean and standard deviation of simulations in each batch.
    
    NOTE:   These are the only features of the simulation needed for later.
            They can and should be stored without taking too much space.

    Parameters
    ----------
    errors : np.ndarray
        errors simulated method in batches and batch simulations

    Returns
    -------
    errors_mean : np.ndarray
        the mean of errors over last index, i.e. batch_simulations
    errors_std : np.ndarray
        the standard deviation over the last index, i.e. batch_simulations
    """
    if strong: errors = errors ** 2
    errors_mean = errors.mean(axis=-1)
    errors_std = errors.std(ddof=1, axis=-1) #NOTE:Redundant? maybe make warning about deviation sizes
    return errors_mean, errors_std

def calc_confidence_intervals(batch_means:np.ndarray, tol=1.5, alpha=None, verbose=False):
    """calculates mean and a `(1 - alpha) * 100 %` confidence interval for the last axis of `error_mean`.
    
    If no alpha is supplied, alpha is chosen so that the error interval in the plot is reasonably large 
     Returns:
     --------
     e : ndarray
        the mean of the batch means in errors_mean
     de : ndarray
        the confidence interval estimate for the said mean 
     """
    # errors_mean: mean of errors in each of the batches
    batches = batch_means.shape[-1]
    e = batch_means.mean(axis=-1)
    error_std = batch_means.std(ddof=1, axis=-1)
    const = error_std / np.sqrt(batches)
    if alpha is None:
        alphas = [1/2, 1/4, 1/5, 1/10, 1/20, 1/100, 1e-3, 1e-4, 1e-5, 1e-6]
        for alpha in reversed(alphas):
            t_stat = scipy.stats.t.ppf(1 - alpha/2, batches)
            de = const * t_stat
            relative_error_deviation = np.max(de[e>1-14] / e[e>1-14])
            print(f"Max value of de/e for alpha={alpha}: {relative_error_deviation}")
            if relative_error_deviation < tol: break
        if verbose: print(f"Accepted {(1-alpha)*100}% confidence interval for plot.")
    else:
        if verbose: print(r"Calculating {}% confidence intervals estimator of mean error of {} batches.".format((1-alpha)*100, batches))
        t_stat = scipy.stats.t.ppf(1 - alpha/2, batches)
        de = const * t_stat    
    return e, de, alpha


def build_from_string(method_name:str, return_method=True):
    """Build HBVM from string of method_name format, i.e. Quadrature-[...](k, s)"""
    quad, rest = method_name.split(sep="-")
    rest, features = rest.split(sep="(")
    quadrature = dict(Gauss=0, Lobatto=3)[quad]
    k, s = eval(features[:-1])
    
    if return_method: return RK(k=k, s=s, quadrature=quadrature)
    else: return quadrature, k, s

def import_simulation(filename:str, relative_path:str, verbose=True):
    """import_simulation_features imports results and parameters of a simulation at "relative_path/filename"

    Collects list of hbvms/rk-methods, `errors`, `error_deviations`, stepsizes `dts` and parameter dictionary `p_dict` from .npz using

    `np.load(f"{relative_path}/{filename}", allow_pickle=True)`
    

    Parameters
    ----------
    filename : str
        name of file where the data is stored as compressed array (.npz file)
    relative_path : str
        the directory of the stored values relative to the current directory

    Returns
    -------
    hbvms : list[RK]
        list of hbvms used in the simulation
    errors : np.ndarray
        batch means of strong and weak errors for the used methods
    error_deviations : np.ndarray
        batch standard deviations of strong and weak errors for the used methods
    dts : np.ndarray
        list of stepsizes
    p_dict : 
        dict of parameters of Single Integrand instance used in simulation
    
    """
    # make full file path string
    if not relative_path: relative_path = "."
    filename_full = f"{relative_path}/{filename}"
    if filename_full[-4:] != ".npz": filename += ".npz"
    
    # Load features from file and store in dictionary:
    file_dict = dict()
    with np.load(filename_full, allow_pickle=True) as data:
        for key, val in data.items(): file_dict[key] = val
    

    # Get method names
    method_names = [*file_dict.keys()][:-3] # the last three are dts, rk_ref and parameters

    # Make method list
    hbvms = [build_from_string(method_name, return_method=True) for method_name in method_names]

    # Get errors and error deviations
    K = len(hbvms)
    error_shape = [K, *file_dict[method_names[0]].shape[1:]] # (K, 2, 2, L), where L = len(dts)
    errors = np.zeros(error_shape)
    error_deviations = np.zeros(error_shape)

    for i in range(K):
        method = method_names[i]
        errors[i], error_deviations[i] = file_dict[method]
    
    # Reorder shape to match the one used in creating convergence plots
    errors = np.moveaxis(errors, 0, 2)
    error_deviations = np.moveaxis(errors, 0, 2)
    
    # make list of dts
    dts:np.ndarray = file_dict['dts']

    # set up dictionary of features/parameters of a SingleIntegrand class / used in experiments:
    p_dict:"dict[str]" = dict(zip(*file_dict['parameters'].T))

    # add features from reference dictionary (rk_ref instance of RK class, power_ref, exact)
    rk_dict = dict(zip(*file_dict['rk_ref'].T))
    p_dict['rk_ref'] = RK(k=rk_dict['k'], s=rk_dict['s'], quadrature=rk_dict['quadrature'])
    p_dict['power_ref'] = rk_dict['power_ref']
    p_dict['quadratures'] = list({hbvm.quadrature for hbvm in hbvms})

    if verbose: print(f"The exact solution was found using {['analytic formula', 'rk_ref approximation'][int(rk_dict['exact'])]}.")

    return hbvms, errors, error_deviations, dts, p_dict


def assess_rk_ref(current, stored):
    # Check if rk_ref of stored  satisfies necessary requirements to be used in this result generation
    if stored["power_ref"]      <   current["power_ref"]:   return False
    elif stored["exact"]        !=  current["exact"]:       return False
    elif stored["s"]            <   current["s"]:           return False # Only need better method
    elif stored["quadrature"]   !=  current["quadrature"]:  return False # Should be based on same quadrature, i think
    else:                                                   return True

def assess_dts(current, stored):
    if np.min(current) < np.min(stored) or np.max(current) > np.max(stored): pass # Possibly better to adjust them
    return len(current) <= len(stored)

def assess_parameters(current, stored):
    # Check if parameters coincide sufficiently
    if (current["mu"] != stored["mu"]) or (current["sigma"] != stored["sigma"]): return False
    elif (current["t0"] < stored["t0"]) or (current["T"] > stored["T"]) or (current["base"] != stored["base"]): return False
    elif (current["seed"] != stored["seed"]) or (current["batches"] > stored["batches"]) or (current["batch_simulations"] > stored["batch_simulations"]): return False
    else: return True

def assess_scale(scale:"str|list[str]"):
    if isinstance(scale, str): # Split into support function
        if scale == "semilogy": scale = ["linear", "log"]
        elif scale == "semilogx": scale = ["log", "linear"]
        elif scale == "loglog": scale = ["log", "log"]
        else: scale = [scale, scale]
    else: pass
    return scale

def get_subplot_dicts(xs, x_ref, errs, H_errs, stochastic, invariant=None, order=1):
    # set up q and p dict sizes
    d = x_ref.shape[-1]//2
    qs, q_ref = [x[...,0] for x in xs], x_ref[...,0]
    ps, p_ref = [x[...,1] for x in xs], x_ref[...,1]
    if d > 2:
        qs, q_ref = [np.linalg.norm(q, ord=order, axis=-1) for q in qs], np.linalg.norm(q_ref, ord=order, axis=-1)
        ps, p_ref = [np.linalg.norm(p, ord=order, axis=-1) for p in ps], np.linalg.norm(p_ref, ord=order, axis=-1)
        q_name, p_name = r"$||q||_{{{}}}$".format(order),  r"$||p||_{{{}}}$".format(order)
    else: q_name, p_name = "q", "p"

    # set up norm dict sizes
    norms, norm_ref = [np.linalg.norm(x, ord=order, axis=-1) for x in xs], np.linalg.norm(x_ref, ord=order, axis=-1)

    # set up invariant/ first integral dict sizes
    if invariant is not None:
        fints = []
        for x in xs:
            fint = jnp.abs(calc_error(x, x[:1], g=invariant, strong=False, global_error=None, use_initial_value=True))
            fints.append(fint)

    ham_title, ode_title = "Hamiltonian Error", "Solution Error"
    subplot_dicts = dict(
        hamiltonian=dict(title=ham_title, ylabel=R"$|H(x_{{exact}})-H(x_{{approx}})|$", scale="semilogy", ys=H_errs, y_exact=None),
        error=dict(title=ode_title, ylabel=r"$||x_{{approx}} - x_{{exact}}||_2$", scale=['linear', 'log'], ys=errs, y_exact=None), 
        q=dict(title="Position q", ylabel=q_name, scale="linear", ys=qs, y_exact=q_ref),
        p=dict(title="Conjugate Momenta p", ylabel=p_name, scale="linear", ys=ps, y_exact=p_ref),
        norm=dict(title=f"{order}-Norm", ylabel=r"$||x||_{{{}}}$".format(order), scale="linear", ys=norms, y_exact=norm_ref),
    )
    if invariant is not None: subplot_dicts["invariant"] = dict(title="Invariant error", ylabel=R"$|G(x_{{exact}}) - G(x_{{approx}})|$", ys=fints, y_exact=None, scale="semilogy")
    return subplot_dicts

def update_subplot_dicts(subplot_list:list, subplot_dicts, xs, x_ref:np.ndarray, default_inputs):
    """Returns cleaned up keys in subplot_list and complementary dicts in subplot_dicts."""
    
    d = x_ref.shape[-1]
    faulty_keys = []
    for i in range(len(subplot_list)):
        key:str = subplot_list[i]
        try: 
            subplot_list[i] = key.lower() # Asserts whether string or somethings else
            subplot_dicts[subplot_list[i]] # Asserts whether valid 
        except Exception:
            if isinstance(key, int) and key < d:
                if d == 2: subplot_list[i] = ["q", "p"][key]
                else: 
                    pass 
                    ys, y_ref = [x[..., key] for x in xs], x_ref[..., key]
                    var = ["q", "p"][key//(d//2)]
                    idx = (key + 1) % (d//2) # index of variable
                    title = ylabel = f"$x_{key+1}$ or ${var}_{idx+1}$"
                    new_key = var+str(idx)
                    subplot_dicts[new_key] = dict(title=title, ylabel=ylabel, ys=ys, y_exact=y_ref, scale="linear")
                    subplot_list[i] = new_key
            else:
                print(f"Faulty subplot specification: {key}.\nChoose from {default_inputs} or integers up to {d-1} for elements of x.")
                faulty_keys.append(key)
        
    for key in faulty_keys: subplot_list.remove(key)
    
    return subplot_list, subplot_dicts