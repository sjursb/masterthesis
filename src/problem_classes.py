# Python script for param classes setting up ode and sde problems.
# Inspired by problem class from project thesis, however not backward compatible yet
import sys
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import param
import scipy.linalg as la

from solver import *
from support_functions import *
from visualization import *

class BaseODE(param.Parameterized): # TODO: Merge with BaseSDE in functionality
    """
    Class giving features of an autonomous ODE problems of the form
    
    1. `x'(t) = f(x), t in [t0, T], x(t0) = x0.`
    
    """
    # Macro
    title       = param.String(default="",  doc="Name of problem")
    
    # Problem features
    x0          = param.Array(default=None,     doc="Initial value(s) of x of dimension d")
    t0          = param.Number(default=0.,      doc="Initial time")
    T           = param.Number(default=1.,      doc="Terminal time")

    H           = param.Callable(default=None, doc="Function H:R^d -> R defining the problem.")
    f           = param.Callable(default=None, doc="A function f:R^d -> R^d, which is the vector field of x derivative in 1.") # TODO: More info on input and output array shape!
    df          = param.Callable(default=None, doc="Jacoubian of vector field f, i.e. df:R^d -> R^(d x d)")
    nu          = param.Integer(None)

    # Special features of problems
    
    hamiltonian = param.Callable(default=None, doc="Hamiltonian function H:R^d -> R of ODEs of the form `x'(t) = J dH(x),` where J is the identity skew-symmetric matrix.")
    invariant   = param.Callable(default=None, doc="Invariant funtion I:R^d -> R of ODEs of the form `x'(t) = S * dI(x),` where S is a skew-symmetric matrix.")
    exact       = param.Callable(default=None, doc="Exact solution of differential equation, if it exists")

    # Variables used generating solution
    base        = param.Selector(default=2., objects=[2.], doc="Base number used in generating stepsizes/timesteps") # Could add more options with faster code
    power_ref   = param.Integer(default=13, bounds=(2, 15), doc="Power used generating reference solution stepsizes/timesteps")
    power_max   = param.Integer(default=11, bounds=(1, 14), doc="Power of smallest stepsize compared; stepsize = 1 / base**power_max.")
    comparisons = param.Integer(default=10, bounds=(0, 14),  doc="Number different step-lengths compared")
    dt_max      = param.Number(default=1., doc="Larges value of timestep which generates workable solution")
    # Solver features
    silent_max = param.Integer(default=7, bounds=(0, 10), doc="Maximum number of silent stages compared with base collocation method")
    quadratures = param.ListSelector(default=[0], objects=[0,3], doc="Which type(s) of quadrature used in HBVMs applied")
    s_range = param.ListSelector(default=[1,2,3], objects=[1,2,3,4,5], doc="Orders of collocation polynomial of base methods compared")
    rk_ref = param.ClassSelector(RK, default=RK(s=5, quadrature=0, numeric=True), doc="RK method used to generate reference solution if no exact expression exists, by default Gauss-6.")

    # Stored features
    x_ref  = param.Array(default=None, doc="Reference solution for problem")


    def __init__(self, **params):
        super().__init__(**params)
        # if self.f is None and self.df is None:
        #     self.get_ode_from_H(self.hamiltonian)
        self.check_power_compatibility(swap=False, verbose=False)
        self.adjust_rk_ref()
        

    @param.depends('base', 'dt_max', 'power_ref', 'power_max', 'comparisons', watch=True)
    def check_power_compatibility(self, swap=False, verbose=True):
        """Asserts that power_ref, power_max and comparisons parameters are compatible and adjusts them accordingly."""
        if self.power_ref <= self.power_max:
            if verbose: print("Reference power (power_ref={}) is larger than power used in analysis (power_max={}).".format(self.power_ref, self.power_max))
            if swap: # Swap values for power_ref and power_max
                temp = self.power_max
                self.power_max = self.power_ref
                self.power_ref = temp
            else: 
                self.power_ref = self.power_max + 2
            if verbose: print("New power_ref: {}. \t New power_max: {}".format(self.power_ref, self.power_max))
        while (self.T - self.t0) < 1 / float(self.base) ** (self.power_max - self.comparisons):
            self.comparisons = self.comparisons - 1
        if verbose: print("New comparisons value: ", self.comparisons)

    @param.depends('s_range', watch=True)
    def adjust_rk_ref(self, extra_stages=2, quadrature=None):
        s_max = int(np.max(self.s_range))
        if quadrature is None: quadrature = self.rk_ref.quadrature
        if s_max + extra_stages != self.rk_ref.s:
            print("Adjusts RK method used to generate reference solution.\nNew reference method: Gauss-{}.".format(2*(s_max + extra_stages)))
            self.rk_ref = RK(s=s_max + extra_stages, quadrature=quadrature)

    @param.depends('base', 'power_ref', 'x0', 'f', 'df', 't0', 'T')
    def generate_reference(self, base_power_ref:"tuple[np.number, np.number]|None"=None, store=True, verbose=True, dW_ref=None):
        """Genereate reference ODE solution either using the exact formula supplied or the high order reference RK method."""
        if base_power_ref is None:
            base, power_ref = self.base, self.power_ref
        else:
            base, power_ref = base_power_ref

        t0 = time()
        if self.exact is None:
            dt = 1 / base ** power_ref
            if verbose: print("Generating ODE reference solution of {} using {} with steplength {}...".format(self.title, self.rk_ref.method_name, dt))
            x_ref = np.array(irk_ode_solver(self.x0, self.rk_ref, self.f, self.df, t0=self.t0, T=self.T, dt=dt))
        else:
            if verbose: print("Generating ODE reference solution of {} using exact expression...".format(self.title))
            timesteps = int((self.T-self.t0) * base ** power_ref)
            t = np.linspace(self.t0, self.T, timesteps + 1)
            x_ref = np.array(self.exact(t))
        if verbose: print("Finished generating ODE reference solution in {} seconds.".format(time()-t0))
        if store: self.x_ref = x_ref
        else: return x_ref

    def get_ode_from_H(self, H:callable=None):
        """Use jax.grad and jax.hessian autodiff functionality to build ODE from supplied Hamiltonian function H(x)."""
        if self.H is None: self.H = H # works on first dimension
        if self.hamiltonian is None: self.hamiltonian = jax.jit(lambda x: self.H(jnp.moveaxis(x, -1, 0))) # works on last dimension
        d = self.x0.shape[0]//2
        
        J = jnp.kron(jnp.array([[0,1],[-1,0]]), jnp.eye(d))
        
        @jax.jit
        def f(x):
            return jnp.dot(J, jax.grad(H)(x))
        
        @jax.jit
        def df(x):
            return jnp.dot(J, jax.hessian(H)(x))
        
        self.f = f; self.df = df

    def ode_solver(self, rk:RK, dt:float, method:str=""):
        """Uses supplied rk method to calculate approximate ODE with irk_ode_solver OR symmetric_solver if `method`name is supplied."""
        if method: 
            return symmetric_solver(self.x0, self.f, self.df, t0=self.t0, T=self.T, dt=dt, method=method)
        else:
            return irk_ode_solver(self.x0, rk, self.f, self.df, t0=self.t0, T=self.T, dt=dt)

    def get_dts(self):
        """Get reference dt and list of dts used in comparison"""
        dt_ref = 1 / self.base ** self.power_ref
        dts = 1 / self.base ** np.arange(self.power_max, self.power_max - self.comparisons, -1)
        return dt_ref, dts
    
    def get_timesteps(self):
        """Get number of timesteps in reference solution and the rest of the computations"""
        dt_ref, dts     = self.get_dts()
        timesteps_ref   = int((self.T-self.t0)/dt_ref)
        timesteps_array = jnp.array((self.T - self.t0)/dts, dtype=int)
        return timesteps_ref, timesteps_array
    

    def get_silent_ranges(self, s_min_max:"tuple[int]"=None) -> "dict[int, range]":
        """Generate ranges of silent stages used for each method to calculate hamiltonian of degree self.nu exactly.
        
         - If non-polynomial Hamiltonian (ie self.nu=None), its automatically set to `self.silent_max` for each value of s.
         - Number of silent stages is also truncated by `self.silent_max`.
         
         Returns
         -------
         silent_ranges : dict[int, range]
            list of silent ranges corresponding to the degree of s used in method
        """
        if s_min_max is not None: 
            s_min, s_max = s_min_max
            s_range = list(range(s_min, s_max+1))
        else:
            s_range = self.s_range
        
        silent_ranges = {}
        for s in s_range:
            r_max = calc_silent_max(self.nu, s, self.silent_max)
            silent_ranges[s] = range(r_max+1)
        return silent_ranges

    def get_hbvms(self, s_min_max=None, quadratures=None, silent_stages=True):
        """Make list of HBVMs used based on problem parameters, s_min_max and/or quadratures, if supplied"""
        if quadratures is None: quadratures = self.quadratures
        if s_min_max is None: s_range = self.s_range
        else:
            s_min, s_max = s_min_max
            s_range = range(s_min, s_max + 1)
            if s_max >= self.rk_ref.s: self.rk_ref = RK(s=s_max + 1)
        # NOTE: creating lists from for loops works from left to right (ie left loop is first or outer loop)
        if silent_stages: silent_ranges:dict = self.get_silent_ranges(s_min_max)
        else: silent_ranges = {s:range(1) for s in s_range} # No silent stages
        hbvms   = [RK(k=s+r, s=s, quadrature=quadrature, numeric=True) for quadrature in quadratures for s in s_range for r in silent_ranges[s]]
        return hbvms
    
    calc_solutions = calc_ode
    calc_hamiltonian = calc_hamiltonian
    
    def plot_error(self, base=None, s_min_max=None, semilog=True, abs_diff=True, hamiltonian=True, use_initial_value=False, show=True):
        """method version of the plot_hamiltonian function from visualization.py"""
        if base is None: base = self.base

        hbvms = self.get_hbvms(s_min_max)
        xs, x_exact = self.calc_solutions(hbvms, base)
        if semilog: scales=["linear", "log"]
        else: scales=["linear", "linear"]

        if hamiltonian:
            Hx, H_exact = self.calc_hamiltonian(xs, x_exact, use_initial_value)
            if abs_diff: 
                Hx = [np.abs(H-H_exact) for H in Hx]
                H_exact = np.zeros(len(H_exact))
                error_text = "Error "
            else: error_text = ""
            title = self.title + r" Hamiltonian ODE {}Plot".format(error_text)
            fig, ax = time_plot(
                title, hbvms, t0=self.t0, T=self.T, ys=Hx, y_exact=H_exact, 
                base=self.base, ylabel="H(x(t))", scales=scales, store=not show
            )
        else:
            errors = la.norm(xs - x_exact, ord=2, axis=-1)
            title = self.title + r" ODE Error Plot"
            fig = time_plot(
                title, hbvms, t0=self.t0, T=self.T, ys=errors, y_exact=np.zeros(len(x_exact)), 
                ylabel=r"$$\lVert x_{{exact}} - x_{{approx}}\rVert_2$", base=self.base, scales=scales, store=not show)
        
        if show: plt.show()
        else: return fig
    
    calc_k_convergence = calc_ode_k_convergence
    k_plot = k_plot

    def plot_k_convergence(self, base_max_ref=None, s_min_max=None, use_initial_value=False, global_error=True):
        if base_max_ref is None: base_max_ref = (self.base, self.power_max, self.power_ref)

        errors, labels = self.calc_k_convergence(base_max_ref=base_max_ref, s_min_max=s_min_max, use_initial_value=use_initial_value, global_error=global_error)
        fig = self.k_plot(errors, labels, base_max_ref=base_max_ref)
        plt.show()

    def calc_convergence(self, hbvms, verbose=False): return calc_convergence(self, hbvms, verbose=verbose, stochastic=False)

    def convergence_plots(self, hbvms=None, s_min_max=None, slopes=None, slope_factor=1e-1):
        """Generates global/local ODE error convergence plot measuresd in hamiltonian/2-norm"""
        
        self.t0, self.T = [0, 1]

        if hbvms is None: hbvms = self.get_hbvms(s_min_max) # Get methods
        dts, e = self.calc_convergence(hbvms, stochastic=False) # Get errors and stepsizes
        if slopes is None: slopes = 2 * np.array(self.s_range) + int(not global_error)
        
        for strong in range(2):
            for global_error in range(2):
                error_type = ["Local", "Global"][global_error]
                error_measure = ["Hamiltonian", "2-Norm"][strong]
                title = self.title + " {} Error Convergence in {} sense".format(error_type, error_measure)
                fig = convergence_plot(title, hbvms, dts, e[strong, global_error], slopes=slopes, slope_factor=slope_factor)
                plt.show()




class BaseSDE(BaseODE):
    """
    Class giving base parameters used in simulating SDEs of the form
    
    2. `dX(t) = f(X)dt + g(X)dW(t), t in [t0, T], E[X(t0)] = x0, dW in R.`

    Essentially what is needed to simulate Wiener Processes.
    """

    # Features used simulating Wiener Processes
    seed                = param.Integer(default=100,    doc="Number used as seed in random number generator")
    batches             = param.Integer(default=8,      doc="Number of batches used in simulations")
    batch_simulations   = param.Integer(default=1250,   doc="Number of Wiener Process simulations in each batch")
    diffusion_terms     = param.Integer(default=0,      doc="Dimension of Wiener proccess (0 gives scalar process).")
    
    # Auxilliary variables
    dW          = param.Array(default=None, doc="Array of Wiener increments")

    # NOTE: wach=True calls method automatically when parameters are changed
    @param.depends('t0', 'T', 'base', 'power_ref', 'batches', 'batch_simulations', 'seed') 
    def generate_dW(self, base_power_ref:"tuple[int,int]"=None, verbose=True):
        """Generate dW Used for reference"""
        if base_power_ref is None: base, power_ref = self.base, self.power_ref
        else: base, power_ref = base_power_ref
        dt = 1 / base ** power_ref
        self.dW:np.ndarray = truncated_wiener(t0=self.t0, T=self.T, dt=dt, batches=self.batches, batch_simulations=self.batch_simulations, seed=self.seed, diffusion_terms=self.diffusion_terms, k=self.rk_ref.k)
        if verbose: print(f"Wiener increments with shape {self.dW.shape} generated.")
    
    def get_dW_list(self, dW=None):
        """Shorthand method for get_dW_list from solver_sde.py."""
        if dW is None:
            if self.dW is None: self.generate_dW()
            dW = self.dW
        return get_dW_list(dW_ref=dW, t0=self.t0, T=self.T, base=self.base, power_max=self.power_max, comparisons=self.comparisons)


class SingleIntegrand(BaseSDE):
    """
    Class for defining single integrand problems, i.e. SDEs of the form 

    3. `dX(t) = f(X) dm = f(X) * (mu * dt + sigma * dW(t)),`

    where the rest of the parameters are as in BaseSDE class.

    Note: sigma is now a single value - array is unnecessary for this problem type.
    """

    # Problem features
    sigma_0      = param.Number(default=1.,  doc="Drift constant in equation 3.")
    sigma   = param.Number(default=1.,  doc="Diffusion constant in equation 3.")

    # Class methods
    def __init__(self, **params):
        super().__init__(**params)
        #self.adjust_comparisons(verbose=False)
    
    @param.depends('sigma', 'batches', 'batch_simulations', 'seed', 'base', 'power_ref', 't0', 'T')
    def generate_dW(self, base_power_ref=None, sigma=None, dt=None, verbose=True): # Note: SingleIntegrand only treats scalar processes!
        if dt is None:
            if base_power_ref is None: base, power_ref = self.base, self.power_ref
            else: base, power_ref = base_power_ref
            dt = 1 / base ** power_ref

        if sigma is None: sigma = self.sigma
        if not sigma: 
            if verbose: print("self.sigma=0 treats the problem as an ODE, i.e. not using batches and batch_simulations")
            timesteps_ref = int( (self.T - self.t0) / dt)
            self.dW = np.zeros((timesteps_ref, 1, 1))
            if verbose: print("dW_ref.shape = {}".format(self.dW.shape))
        else:
            self.dW:np.ndarray = truncated_wiener(t0=self.t0, T=self.T, dt=dt, batches=self.batches, batch_simulations=self.batch_simulations, seed=self.seed, diffusion_terms=0, k=self.rk_ref.k)
            if verbose: print(f"Wiener increments with shape {self.dW.shape} and size {HumanBytes.format(self.dW.nbytes)}")

    @param.depends('base', 'power_ref', 't0', 'T', 'batches', 'batch_simulations', 'x0', 'rk_ref')
    def generate_reference(self, base_power_ref:"tuple[int,int]"=None, dW_ref=None, stochastic=True, verbose=True):
        """Generate reference SDE solution using irk_sde_solver - NB: replaces method from BaseODE!"""
        if base_power_ref is None: base, power_ref = self.base, self.power_ref
        else: base, power_ref = base_power_ref
        
        if dW_ref is None: 
            if stochastic: sigma=self.sigma
            else: sigma=0
            self.generate_dW(base_power_ref, sigma=sigma)
        else: 
            self.dW = dW_ref
            dW_ref = None # To avoid using unnecessary memory

        # Do the computation
        time0 = time()
        self.x_ref = self.calc_reference(self.t0, self.T, self.dW, verbose=verbose)
        if verbose: 
            print("Finished generating reference in {} seconds.".format(time()-time0))
            print("Reference solution has shape", self.x_ref.shape)

    def calc_reference(self, t0, T, dW_ref, sigma=None, verbose=True):
        """Calculate reference solution over interval `[t0, T]` using simulated WP given by dW_ref of shape (timesteps, batches, batch_simulations)."""
        timesteps = self.dW.shape[0]
        dt = int((T-t0)/timesteps)
        if sigma is None: sigma=self.sigma
        if self.exact is None:
            if verbose: print(f"Calculating reference solution of {self.title} using {self.rk_ref.method_name} with stepsize {self.base}**(-{self.power_ref}).")
            reference = self.sde_solver(self.rk_ref, dt, dW=dW_ref, verbose=verbose)
        else:
            if verbose: print("Calculating reference solution of {} using exact formula...".format(self.title))
            reference = calc_exact_solution(self.exact, t0, T, timesteps, sigma_0=self.sigma_0, sigma=sigma, dW=dW_ref)
            if not self.sigma: reference.squeeze()
            #FIXME: Be mindful of exact solution and initial values!!
        return reference
    

    @param.depends('comparisons', 'base', 'power_max', 'dt_max', watch=False)
    def adjust_comparisons(self, verbose=True): #TODO: Rewrite or remove if nan feature implemented in Newton iterations
        if 1 / float(self.base) ** (self.power_max - self.comparisons) > self.dt_max:
            self.comparisons = int((self.power_max * np.log(self.base) + np.log(self.dt_max))/np.log(self.base))
            if self.comparisons < 5: 
                self.power_max += (5 - self.comparisons) # At least five comparisons is nice.
                self.comparisons = 5
            if verbose: 
                print(f"Too large max stepsize! new values: comparisons={self.comparisons}, power_max={self.power_max}, power_ref={self.power_ref}")

    def sde_solver(self, rk:RK, dt, dW:"None|np.ndarray"=None, t0=None, T=None, sigma=None, verbose=True, backend='cpu', ):
        """Shorthand method for irk_sde_solver only taking parameters rk, dt, dW(optional) and verbose(True/False)."""
        
        if sigma is None: sigma = self.sigma
        if dW is None: 
            if self.dW is None: self.generate_dW(sigma=sigma, dt=dt)
            dW = self.dW
        if t0 is None: t0 = self.t0
        if T is None: T = self.T
        
        return irk_sde_solver(self.x0, rk, self.f, self.df, sigma_0=self.sigma_0, sigma=sigma, dW=dW, t0=t0, T=T, backend=backend)


    def sde_solver_old(self, rk:RK, dt, dW:"None|np.ndarray"=None, t0=None, T=None, sigma=None, verbose=True):
        """Shorthand method for irk_sde_solver_old only taking parameters rk, dt, dW(optional) and verbose(True/False)."""
        
        if sigma is None: sigma = self.sigma
        if dW is None: 
            if self.dW is None: self.generate_dW(sigma=sigma, dt=dt)
            dW = self.dW
        if t0 is None: t0 = self.t0
        if T is None: T = self.T
        return irk_sde_solver_old(self.x0, rk, self.f, self.df, sigma_0=self.sigma_0, sigma=sigma, dW=dW, t0=t0, T=T)

    calc_solutions = calc_sde
    
    def get_parameters(self, rk_ref=False, use_exact=True): 
        """returns array of parameter keys and values stored in problem data dictionary generated by store_computed_errors."""
        if rk_ref: 
            if use_exact: exact = self.exact is not None
            else: exact = 0
            keys = ["k", "s", "quadrature", "power_ref", "exact"]
            values = [self.rk_ref.k, self.rk_ref.s, self.rk_ref.quadrature, self.power_ref, exact]
        else: 
            keys=["sigma_0", "sigma", "x0", "t0", "T", "base", "seed", "batches", "batch_simulations"]
            values=[self.sigma_0, self.sigma, self.x0, self.t0, self.T, self.base, self.seed, self.batches, self.batch_simulations]
        
        return np.array([[key, value] for key, value in zip(keys, values)], dtype=object)

    
    def store_computed_errors(self,
            hbvms:"list[RK]", dts, errors, deviations, path=".", use_exact=True, verbose=True
        ):
        """Store computed errors, dts, stds and other features of the problem in a npz file for later access.
        
        Hopefully, this will save some future calculations...
        """
        
        #NOTE: This might work for dts=t and errors=errors(t), too!
        # Set up array dictionary to input into np.savez_compressed 
        file_dict:"dict[str, np.ndarray]" = dict(zip([hbvm.method_name for hbvm in hbvms], [[errors[:,:,k], deviations[:,:,k]] for k in range(len(hbvms))]))
        file_dict['dts'] = dts
        file_dict['rk_ref'] = self.get_parameters(rk_ref=True, use_exact=use_exact)
        file_dict['parameters'] = self.get_parameters()
        filename = f"{path}/{self.title}_sigma_{self.sigma}_v"

        # Get new filename to avoid overwriting old files
        for i in range(100): 
            temp_filename = f"{filename}{i//10}{i%10}"
            if not os.path.exists(temp_filename + ".npz"): break
        
        # Seal the deal
        np.savez_compressed(temp_filename, **file_dict)
        if verbose: print(f"Saved experiment results in file '{temp_filename}.npz'.")
        return None
    
    def adjust_parameters(self, **kwargs):
        """Adjust class parameters by entering keyword arguments"""
        for key, value in kwargs.items(): setattr(self, key, value)


    def calc_convergence(self, hbvms=None, s_min_max=None, dW_ref=None, tol=2.0, verbose=True, path="", filename="", alpha=None, mask=False, stochastic=True):
        """Calculates errors and confidence intervals (or collects them from file) for specified configuration (hamiltonian, global and alpha)."""
        if filename:
            hbvms, errors, error_deviations, dts, p_dict = import_simulation(filename, path)
            self.adjust_parameters(**p_dict)
            if verbose: 
                print(
                    f"\nImported simulation '{path}/{filename}' with parameters:", *[f"{key}={value}" for key, value in p_dict.items()], 
                    "\nAnd methods", *[hbvm.method_name for hbvm in hbvms],
                    sep="\n"
                )
            if p_dict['sigma']: stochastic = True
            else: stochastic = False
        else:
            self.t0, self.T = [0, 1]
            if hbvms is None: hbvms = self.get_hbvms()
            # Adjust rk_ref to match max k
            k_max = np.max([hbvm.k for hbvm in hbvms])
            if k_max > self.rk_ref.k: self.adjust_rk_ref(extra_stages=int(k_max))
            dts, errors, error_deviations = calc_convergence(self, hbvms, dW_ref=dW_ref, verbose=verbose, stochastic=stochastic, path=path)
        if stochastic:
            e, de, alpha = calc_confidence_intervals(errors, tol=tol, alpha=alpha, verbose=verbose) # mean of (mean of batches) and its standard deviation
            if mask:
                e = np.ma.masked_where(tol*e<de, e)
                de = np.ma.masked_where(tol*e<de, de)
        else: e, de, alpha = errors, error_deviations, 0
        return hbvms, dts, e, de, alpha
    
    def calc_sde_convergence(self, hbvms, dW_ref=None, tol=1, verbose=True):
        return self.calc_convergence(self, hbvms, dW_ref=dW_ref, tol=tol, verbose=verbose, stochastic=True)
    
    def relative_convergence_plot(self, hbvms:"list[RK]"=None, s_min_max=None, quadratures=[0,3], dW_ref:np.ndarray=None, 
            stochastic=False, store=False,  verbose=True):
        """Calculate relative strong/ODE convergence order for different methods of same theoretical order."""
        self.t0, self.T = 0, 1
        if s_min_max is None: 
            s_range = self.s_range
        else: 
            s_min, s_max = s_min_max
            s_range = range(s_min, s_max + 1)
        
        # Get methods
        if hbvms is None: hbvms = self.get_hbvms(s_min_max, quadratures=quadratures)
        hbvm_refs = self.get_hbvms(s_min_max, quadratures=[0], silent_stages=False) # reference collocation method

        # Calculate convergence values
        hbvms, dts, e_ref, de_ref, alpha = self.calc_convergence(hbvm_refs, dW_ref=dW_ref, stochastic=stochastic)
        _, e, de, _ = self.calc_convergence(hbvms, dW_ref=dW_ref, stochastic=stochastic)

        # Compare convergence order
        relative_errors = calc_relative_convergence(hbvms, e, hbvm_refs, e_ref)

        # Set up plot
        title = f"{self.title} relative convergence order"
        
        fig = relative_convergence_plot(title, hbvms, dts, relative_errors, s_range, store=store)

        plt.show()

    def convergence_plots(self, hbvms=None, s_min_max=None, slopes=None, slope_factor=1e-1, path="", tol=2.0, stochastic=True, verbose=False):
        """Generates global/local SDE error convergence plot measured in hamiltonian/2-norm"""
        # Get quantities to make plot
        hbvms, dts, e, de, alpha = self.calc_convergence(hbvms, s_min_max=s_min_max, path=path, stochastic=stochastic, tol=tol, verbose=verbose)
        if slopes is None: slopes = get_slopes(hbvms, self.nu, stochastic=stochastic)
        else: pass
        
        ylabels = [
            [R"$|H(x_{{approx}})-H(x_{{exact}})|$", R"$||x_{{exact}} - x_{{approx}} ||_2$"],
            [R"$\left|E\left[H(x_{{approx}})\right] - E\left[H(x_{{exact}})\right]\right|$", R"$\sqrt{{E\left[||x_{{exact}} - x_{{approx}} ||_2^2\right]}}$"]
        ][stochastic]

        for strong in range(2):
            for global_error in range(2):
                if global_error: error_type = ["Global", ""][stochastic]
                else: error_type = "Local"
                if strong: error_measure = ["Solution", "Strong Sense"][stochastic] 
                else: error_measure = ["Hamiltonian", "Weak/Hamiltonian Sense"][stochastic]
                
                title = self.title + " {} Error Convergence in {}".format(error_type, error_measure)
                if alpha:
                    title += f", {np.round((1-alpha)*100, 4)}% Confidence"
                plot_slopes = slopes[strong]
                if len(plot_slopes): plot_slopes + int(not global_error) # one extra for local error
                fig = convergence_plot(title, hbvms, dts, e=e[strong, global_error], de=de[strong, global_error], slopes=plot_slopes, slope_factor=slope_factor, ylabel=ylabels[strong])
                plt.show()
    
    def double_convergence_plots(self, 
            hbvms=None, s_min_max=None, figure_path="", table_path="", data_path="",
            max_slope=8, slope_factor=None, tol=2.0, stochastic=True, simulation_filename="", alpha=None,
            verbose=False, store=False, mask=False, order_tol=1e-14): 
        """Generates global and local error convergence plots with Hamiltonian/Weak and ODE/Strong error subplots."""
        
        # To avoid the approximations to blow up due to unstable choices of stepsizes
        if stochastic: self.adjust_comparisons()

        # Get quantities to make plot
        hbvms, dts, e, de, alpha = self.calc_convergence(hbvms=hbvms, s_min_max=s_min_max, path=data_path, stochastic=stochastic, alpha=alpha, tol=tol, filename=simulation_filename, verbose=verbose, mask=mask)

        for global_error in range(2):
            if global_error: error_type = [" Global", ""][stochastic]
            else: error_type = [" Local", ""][stochastic]
            
            title = f"{self.title}{error_type} Error Convergence"
            if alpha:
                title += f", {np.round((1-alpha)*100, 4)}% Confidence"

            quadrature = self.quadratures[0]
            filename = title + [" deterministic", " stochastic"][stochastic] +{0:" gauss", 3:" lobatto"}[quadrature] 

            if global_error: 
                if store:
                    store_table_path = table_path # TODO: Rewrite to remove redundancy
                    if not len(store_table_path): store_table_path = "./"
                else: store_table_path = ""

                order_table = get_order_table(
                    hbvms, e[:, global_error], stochastic, global_error, order_tol=order_tol,
                    nu=self.nu, base=self.base, filename=f"{filename} order table", store_table_path=store_table_path
                )
                print(f"Order table for {self.title}:\n", order_table)
            if slope_factor is None: slope_factor = 1e-1
            slopes = get_slopes(hbvms, self.nu, stochastic=stochastic)
            fig = double_convergence_plot(title, hbvms, dts, e=e[:, global_error], de=de[:, global_error], max_slope=max_slope, slope_factor=slope_factor, slopes=slopes, global_error=global_error, store=store, stochastic=stochastic)
            if store:
                fig_filename_full = f"{figure_path}/{filename}.pdf".replace("%", "")
                plt.savefig(fig_filename_full, bbox_inches='tight')
                if verbose: print(f"Saved figure at '{fig_filename_full}'")
            
            
            else: plt.show() # Assumes you adjust everything manually


    def long_time_solutions(self, hbvms:"list[RK]"=None, dts:"list[float]"=None, dt=None, t0=0, T=1e3, verbose=True, use_init_value=True, stochastic=True):
        if hbvms is None: hbvms = self.get_hbvms()
        K = len(hbvms)
        if dts is None: 
            if dt is None: dt = 1 / self.base ** self.power_max
            dts = [dt for _ in range(K)]
        else: pass
        self.t0, self.T = t0, T
        dt_ref = 1 / self.base ** self.power_ref
        timesteps_ref = int((T-t0)/dt_ref)
        
        if not stochastic: 
            sigma=0.
            self.dW = np.zeros((timesteps_ref, 1, 1))
        else: 
            sigma = self.sigma
            k_min = np.min([hbvm.s for hbvm in hbvms])
            self.dW = truncated_wiener(t0, T, dt_ref, batches=1, batch_simulations=1, seed=self.seed, k=k_min)
        
        # Calculate reference
        time0 = time()
        x_ref = self.calc_reference(t0, T, self.dW, sigma=sigma, verbose=verbose).squeeze()
        if verbose: print(f"Finished generating reference in {np.round(time()-time0, 2)} seconds.")
        
        # Calculate Hamiltonian
        if use_init_value: H_ref = np.full(x_ref.shape[:-1], self.hamiltonian(self.x0))
        else: H_ref:np.ndarray = self.hamiltonian(x_ref)
        
        xs = []; H_errs = []; errs = []

        # Calculate solutions for methods
        for hbvm, dt in zip(hbvms, dts):
            step = int(dt/dt_ref) #NOTE: assumes dts are negative powers of self.base!
            temp_shape = (self.dW.shape[0]//step, step, 1, 1) # timesteps_ref = self.dW.shape[0]
            dW_temp = self.dW.copy().reshape(temp_shape).sum(axis=1)
            
            time0 = time()
            dW_temp.shape
            if verbose: print(f"\nCalculating solution using {hbvm.method_name} with {int((T-t0)/dt)} timesteps...")
            x = self.sde_solver(hbvm, dt, dW_temp, t0=t0, T=T, sigma=sigma, verbose=verbose).squeeze()
            if verbose: print(f"Finished in {np.round(time() - time0, decimals=2)} seconds.")
            
            err = calc_error(x, x_ref[::step], g=self.hamiltonian, strong=True, global_error=None)
            H_err = jnp.abs(calc_error(x, x_ref[::step], g=self.hamiltonian, strong=False, global_error=None, use_initial_value=use_init_value))

            xs.append(x); errs.append(err), H_errs.append(H_err)
        
        return xs, x_ref, errs, H_errs, H_ref

    def time_plots(self, 
            hbvms, dts:"list[float]"=None, dt:float=None, t0=0, T=1e3, subplot_list:"list[str]"=None, axs:"list[plt.Axes]"=None, order=1, legend_idx=0, values=None,
            use_init_value=True, stochastic=True, suptitle="", store=False, verbose=True, show=True
        ):
        """Generates plots of different features of solutions generated hbvms with time as common x-axis."""
        default_inputs = ["hamiltonian", "error", "q", "p", "norm"]
        if self.invariant is not None: default_inputs.append("invariant")
        if subplot_list is None: subplot_list=default_inputs
        
        # Calculation
        if values is None: values = self.long_time_solutions(hbvms, dts, dt, t0, T, verbose, use_init_value, stochastic=stochastic)
        else: pass
        xs, x_ref, errs, H_errs, *_ = values

        # Prepare plotting
        subplot_dicts = get_subplot_dicts(xs, x_ref, errs, H_errs, stochastic, order=order, invariant=self.invariant)
        subplot_list, subplot_dicts = update_subplot_dicts(subplot_list, subplot_dicts, xs, x_ref, default_inputs) # remove bad subplot names
        
        # Do the plotting
        n_plots = len(subplot_list)
        if axs is None:
            fig, axs = plt.subplots(n_plots, 1, sharex=True)
            if not suptitle: suptitle = self.title
            if np.allclose(dts, np.full_like(dts, dts[0])): suptitle += R", $\Delta t = {}^{{-{}}}$".format(int(self.base), -int(np.log(dts[0])/np.log(self.base)))
            fig.suptitle(suptitle)
        else:
            fig = None
        for i in range(n_plots):
            if i == legend_idx or i == n_plots + legend_idx: show_legend=True
            else: show_legend=False
            time_plot(hbvms, t0, T, ax=axs[i], base=self.base, show_legend=show_legend, **subplot_dicts[subplot_list[i]])
            
        # plt.figlegend() # This one needs configuration to be pretty
        #FIXME: add smart way to take legends from a given axis and input into figlegend with nice placement.
        if store:
            pass # plt.savefig(fname.pdf, ...)
        else: pass
        if show: plt.show()
        else: return fig, axs
        
    def H_contour_plot(self, H, 
            hbvms:"list[RK]"=None, dts=None, dt=None, t0=0, T=1e3, stochastic=True, use_init_value=True,
            levels=21, time_plots:"list[str]|bool"=False, alpha=.5, show=False, xlim=None, ylim=None, title=None,
            verbose=True,  colormap='viridis', store=False, show_legend=False
        ):
        """Contour plot of H with solutions generated by hbvms plotted onto it."""
        # TODO: Separate out time_plots and rather make a compound function afterwards
        # TODO: Make contour plot function class Parameter H and only make available to problems with this supplied!
        if hbvms is None: hbvms = self.get_hbvms()
        # Calculate intermediary results
        if time_plots: xs, *_ = self.long_time_solutions(hbvms=hbvms, dts=dts, dt=dt, t0=t0, T=T, stochastic=stochastic, use_init_value=use_init_value, verbose=verbose)
        else:
            if dts is None: dts = [dt]* len(hbvms)
            xs = []
            dt_ref = np.min(dts)
            if not stochastic: 
                sigma=0.
                timesteps_ref = int((T-t0)/dt_ref)
                self.dW = np.zeros((timesteps_ref, 1, 1))
            else: 
                sigma = self.sigma
                k_min = np.min([hbvm.s for hbvm in hbvms])
                self.dW = truncated_wiener(t0, T, dt_ref, batches=1, batch_simulations=1, seed=self.seed, k=k_min)
            for dt, hbvm in zip(dts, hbvms):
                
                x = self.sde_solver(hbvm, dt, sigma=sigma, t0=t0, T=T, verbose=verbose).squeeze()
                xs.append(x)

        # make contour figure
        if time_plots is False:
            fig, ax1 = plt.subplots(dpi=128) # TODO: Make figsize and dpi user adjustable/match problem
            right = 1.2
            ncol = 2
        else:
            if time_plots is True: time_plots = ["hamiltonian", "error"]
            n = len(time_plots)
            fig = plt.figure(figsize=(10,5), dpi=333)
            ax1 = plt.subplot2grid((n, 2), (0, 0), rowspan=n)
            axs = [plt.subplot2grid((n,2), (i, 1)) for i in range(n)]
            right=2.5
            ncol=2

        
        hbvm_colors = make_color_lists(hbvms)
        if np.allclose(dts, np.full_like(dts, dts[0])): # If only one stepsize
            label_extra = ["" for _ in range(len(hbvms))]
            title_extra = R", $\Delta t = {}^{{{}}}$".format(int(self.base), int(np.log(dts[0])/np.log(self.base)))
        else: # For different stepsizes
            title_extra = ""
            label_extra = [R", $\Delta t = {}^{{{}}}$".format(int(self.base), int(np.log(dts[i])/np.log(self.base))) for i in range(len(hbvms))]
        line_list = [
            [
                np.moveaxis(xs[i], -1, 0), 
                dict(
                    color=hbvm_colors[i], 
                    label=hbvms[i].method_name+label_extra[i],
                    marker='.', markersize=1., 
                )
            ] for i in range(len(hbvms))
        ]

        q_values = np.array([[np.min(x[...,0]), np.max(x[...,0])] for x in xs])
        p_values = np.array([[np.min(x[...,1]), np.max(x[...,1])] for x in xs])
        
        if xlim is None:
            xlim = np.array([np.min(q_values[...,0]), np.max(q_values[...,1])])
            xlim = xlim + alpha * np.abs(xlim) * np.array([-1, 1])
        if ylim is None:
            ylim = np.array([np.min(p_values[...,0]), np.max(p_values[...,1])])
            ylim = ylim + alpha * np.abs(ylim) * np.array([-1, 1])
        H0 = self.hamiltonian(self.x0)
        zlim = H0 + np.array([-alpha, alpha])
        contour_title = f"Hamiltonian contours, $T=10^{int(np.log10(T))}$" + title_extra
        contour_dict = dict(title=contour_title, xlabel="q", ylabel="p", xlim=xlim, ylim=ylim, aspect="equal")
        # TODO: Option to use different initial values?
        if time_plots:
            values = xs, *_
            _, axs = self.time_plots(hbvms, dts, dt, t0, T, time_plots, axs=axs, legend_idx=n+1, values=values, stochastic=stochastic, verbose=verbose, show=False)

        cmap = plt.get_cmap(colormap)
        fig, ax1 = contour_plot(fig, ax1, H, f_name="H(x)", plot_dict=contour_dict, n_levels=levels, lines=line_list, zlim=zlim, cmap=cmap)
        
        if show_legend: ax1.legend(loc="upper left", bbox_to_anchor=(0., -0.1, right, -0.1), ncol=ncol, mode="expand", borderaxespad=0.)

        fig.set_tight_layout(True)
        if store: 
            plt.tight_layout()
            plt.savefig(f"{self.title} Hamiltonian contours {['deterministic', 'stochastic'][stochastic]}.png")
            #TODO: Set file type matching the problem - H6 needs .png to be correct...
        if show: plt.show()
        else: return fig










################### ------------ end class ----------- #########################
        

        





        
    


# NOTE: Look at this only when you have time
class AdditiveDrift(BaseSDE):
    """
    Class for defining Hamiltonian problems with additive drift, i.e. SDEs of the form

    4. `dX(t) = J dH(X) dt + Sigma dW(t),`

    where Sigma is a (d x m) drift matrix and the deterministic problem is a Hamiltonian ODE. 
    """

    sigma = param.Array(doc="Diffusion matrix.", default=np.array([0.1]))
    


# TODO: Make class for more general SDE problems -> see project thesis code

##################### PROJECT THESIS CODE #########################################
# NOTE: Define good classes ensuring backward compatibility with project thesis code. 

# class BaseClass(BaseSDE):
#     oscillator   = param.Boolean(default=False, doc="Wether or not the problem is a stochastic harmonic oscillator, allowing the use of symMi scheme.")
    
#     # Problem defining variables
#     stepsizes   = param.Array(default=np.linspace(.1,1.0, num=9)), #default=2. ** (- np.arange(2,11)) # Somehow doesn't work with np.array as input variable
#     runs        = param.Integer(default=10**1)

#     # Additional problem features
#     g           = param.Callable(default=None, doc="m-dimensional function g: R^d -> R^(d x m)")
#     dg          = param.Callable(default=None, doc="derivative of function g (Note: This is not constant matrix!)") # CZECH!

#     # Features of the solution and its distribution
#     expected    = param.Callable(doc="First moment of exact solution (if available)")
#     mom2        = param.Callable(doc="Second moment of exact solution (if available)")
#     invariant   = param.Callable(doc="Invariant function of given problem")



#     # Some methods for this class 
    
#     def __init__(self, **params):
#         """Initializes the class and constructing implicit variables"""
#         super().__init__(**params) # Basic initialization

#         # Initialize simulated Wiener Process
#         if self.dW is None:
#             self.generate_new_wiener()
    
    
#     def kwargs(self, dW=None, theta=None, dt=1e-2):
#         """Makes dictionary of input to generic solver"""
#         d = dict(x0=self.x0, dt=dt, t0=self.t0, T=self.T, g=self.g, dg=self.dg, m=self.m, runs=self.runs, dW=self.dW, seed=self.seed)
#         if theta is not None:
#             d['theta'] = theta
#         if dW is not None:
#             d['dW'] = dW
#         return d

#     @param.depends('runs', 'seed', 'stepsizes', 't0', 'T', watch=True)
#     def generate_new_wiener(self):
#         """
#         Builds Wiener process increment matrix of dimensions (N x m x runs) and stores it in class dW.
        
#         Uses smallest stepsize in self.stepsizes as base.
#         """
#         np.random.seed(self.seed)
#         dt = np.min(self.stepsizes)
#         N = int((self.T - self.t0)/dt)
#         self.dW = np.sqrt(dt) * np.random.normal(size=(N, self.m, self.runs))
#         return None


if __name__ == "__main__":
    
    if False:
        BaseODE()
        BaseSDE()
        SingleIntegrand(dW=np.arange(10), sigma=np.arange(4).reshape(2,2))
        AdditiveDrift()
        # BaseClass(stepsizes=1./np.arange(1,5))
        print("Successfully set up all classes.")
    
            
        
