# jax implementation of solvers used in master, which is a lot quicker than numpy, but only runs on linux/wsl or mac

import os

import numpy as np
import scipy.linalg as la

# To use cpu cores in calculations:
from local_variables import xla_flag
os.environ['XLA_FLAGS'] = xla_flag # max 8 on my laptop, max 28 on markov (weird number)

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit

from rk_builder import RK

jax.config.update("jax_enable_x64", True) # makes it possible to use float64 for calculations

# Standard numpy construction of truncated_wiener 
def truncated_wiener(
        t0=0, T=1, dt=1e-1, 
        batches=8, batch_simulations=125, seed=100,
        diffusion_terms=0, k=2, truncate=True
    ):
    """truncated_wiener simulates Wiener Process increments (optionally truncated) using numpy implementation.

    Generates array of truncated variables sampled from a normal distribution with each element a ~ N(0, dt).
    The dimension of the array is (timesteps, batches, batch_simulations, diffusion_terms), where
    
        timesteps = (T - t0) / dt
    
    and the rest are keyword arguments. The truncation bound is given by formula

        A_h = sqrt(2 * k * |log(dt)|)
    
    and the random number generator used is PCG64 (default in numpy as of 2022).
    
    The truncation is based on Stochastic Numerics in Mathematical Physics by Milstein & Tretyakov,
    where any increment absolutely larger is replaced by A_h with the same sign.

    Parameters
    ----------
    t0 : int, optional
        start time, by default 0
    T : int, optional
        stop time, by default 1
    dt : float, optional
        stepsize, by default 1e-1
    batches : int, optional
        Number of simulated batches, by default 8
    batch_simulations : int, optional
        Number of simulated Wiener Processes in each batch, by default 125
    seed : int, optional
        Random generator seed, by default 100
    diffusion_terms : int, optional
        number of diffusion terms for each simulated paths, by default 1
    k : int, optional
        constant in truncation bound (typically the order of the method), by default 2
    truncate : bool, optional
        decides wether to truncate simulated wiener increments or not, by default True

    Returns
    -------
    dW : np.ndarray
        simulated Wiener Proces increments, possibly truncated
    """
    # Initialize empty array
    timesteps = int((T-t0)/dt)
    noise_shape = [timesteps, batches, batch_simulations]

    if diffusion_terms: noise_shape.append(diffusion_terms)
    # Set up random number generator
    rng = np.random.Generator(np.random.PCG64(seed=seed)) # Same as np.random.default_rng()

    # Generate normally distributed variables
    dW = np.sqrt(dt) * rng.standard_normal(noise_shape)
    
    if truncate:
        A_h = np.sqrt(2*k*np.abs(np.log(dt))) # Set bound A_h

        # Set overriding values of dW to +-A_h
        dW[dW> A_h] = A_h
        dW[dW< -A_h] = -A_h
    
    return dW

# jax construction of truncated wiener simulation (redundant, but illustrative of differing syntax)
@jit
def truncate_wiener(dW:jnp.ndarray, k:int, dt:jnp.float64):
    """
    truncate_wiener truncates supplied array using jax functions

    The truncation essentially works the same way as in truncated_wiener,
    with any values absolutely larger than A_h, given by the formula
    
    A_h = sqrt(2 * k * |log(dt)|)

    replaced by A_h with the same sign.

    Parameters
    ----------
    dW : jnp.ndarray
        simulated Wiener increments as DeviceArray (jax implementation of arrays)
    k : int
        constant in truncation bound 
    dt : jnp.float64
        temporal stepsize

    Returns
    -------
    jnp.ndarray
        truncated version of dW
    """
    A_h = jnp.sqrt(2*k*jnp.abs(jnp.log(dt)))
    # Truncating variable
    return jnp.where(jnp.abs(dW) < A_h, dW, jnp.sign(dW) * A_h)

def generate_wiener_jax(dt:jnp.float64, seed:int, timesteps:int, batches:int, batch_simulations:int, diffusion_terms=0):
    """
    generate_wiener_jax generates Wiener Process increments using jax functions.

    Parameters
    ----------
    dt : float
        temporal stepsize
    seed : int
        seed of Random Number Generator
    timesteps : int
        number of timesteps over which a Wiener Process is simulated
    batches : int
        number of simulation batches
    batch_simulations : int
        number of simulations in each batch
    diffusion_terms : int, optional
        dimension of the Wiener Process (zero for scalar Wiener Process), by default 0

    Returns
    -------
    jnp.array
        simulated Wiener Process increments array 
    """
    noise_shape = [timesteps, batches, batch_simulations]
    if diffusion_terms: noise_shape.append(diffusion_terms)
    key = jax.random.PRNGKey(seed)
    return jnp.sqrt(dt) * jax.random.normal(key, shape=noise_shape)


def truncated_wiener_jax(
        t0=jnp.float64(0), T=jnp.float64(1), dt=jnp.float64(1e-1), 
        batches=8, batch_simulations=125, seed=100, 
        diffusion_terms=0, k=2, truncate=True):
    """
    truncated_wiener_jax is anologous to truncated_wiener, but uses jax functions internally.

    Parameters
    ----------
    t0 : jnp.float64, optional
        initial time, by default jnp.float64(0)
    T : jnp.float64, optional
        terminal time, by default jnp.float64(1)
    dt : jnp.float64, optional
        steplength, by default jnp.float64(1e-1)
    batches : int, optional
        number of batches of simulations, by default 8
    batch_simulations : int, optional
        number of simulations in each batch, by default 125
    seed : int, optional
        seed of Random Number Generator, by default 100
    diffusion_terms : int, optional
        dimension of simulated Wiener Process (zero for scalar processes), by default 0
    k : int, optional
        constant in truncation bound, by default 2
    truncate : bool, optional
        whether to truncate the simulateed Wiener Increments, by default True

    Returns
    -------
    jnp.array
        simulated Wiener Process increment, possibly truncated
    """
    
    timesteps = int((T-t0)/dt)
    dW = generate_wiener_jax(dt, seed, timesteps, batches, batch_simulations, diffusion_terms=diffusion_terms)
    
    if truncate: return truncate_wiener(dW, k, dt)
    else: return dW

# Support sizes for the jaxed irk_sde_sovler

class RK1():
        """Temporary class which stores A, b, and c from RK_butcher as jnp.ndarrays (param class in rk_builder only allows for np.array for standard parameters)."""
        def __init__(self, rk:RK):
            self.A = jnp.float64(rk.A)
            self.b = jnp.float64(rk.b)
            self.c = jnp.float64(rk.c)

def F(Z:jnp.ndarray, z0:jnp.ndarray, f:function):
    """
    F evaluates the vector field f at the approximations of x_new, i.e. f(z0 + z) for z in Z (see Thesis pp.52).

    Parameters
    ----------
    Z : jnp.array
        vector of stage approximations of x_new - z0
    z0 : jnp.array
        old value of x (i.e. initial value of Newton iterations)
    f : function
        vector field function of differential equation

    Returns
    -------
    jnp.array
        the resulting vector, where each element is given as f(y0 + y) for y in Y.
    """
    return jax.vmap(lambda z: f(z0 + z))(Z)
    
def calc_gamma(z0:jnp.ndarray, dm:jnp.float64, I_d:jnp.ndarray, A:jnp.ndarray, df:function):
    """
    calc_gamma calculates the left-hand side of the linear systems in the Newton iterations as described in Thesis at page 52

    Calculates and returns the left-hand side matrix of linear system of the Newton iterations (see Thesis pp. 52, step four).

    Parameters
    ----------
    z0 : jnp.ndarray
        starting value of the Newton iterations (or former value of the integrated variable)
    dm : jnp.float64
        current time increment of the modified measure dm = sigma_0 * dt + sigma * dW
    I_d : jnp.ndarray
        Identity matrix of dimension (stages * d) x (stages * d)
    A : jnp.ndarray
        Coefficient matrix of the applied IRK method
    df : function
        Jacobian of the vector field function f

    Returns
    -------
    jnp.ndarray
        matrix of linear system solved in each step of the Newton iterations.
    """
    return I_d - dm * jnp.kron(A, df(z0))

def G(Z:jnp.ndarray, z0:jnp.ndarray, dm:jnp.float64, A_d:jnp.ndarray, F:function):
    """
    G corresponds to the function on the right-hand side of the Newton iterations (Thesis pp. 52, step four).

    Parameters
    ----------
    Z : jnp.ndarray
        vector of stage approximations subtracted the previous timestep
    z0 : jnp.ndarray
        initial value of the Newton iterations, i.e. the previous timestep
    dm : jnp.float64
        current time increment of the modified measure dm = sigma_0 * dt + sigma * dW
    A_d : jnp.ndarray
        Kronecker product of A coefficient matrix and identity matrix of dimension (d x d)
    F : function
        see function definition of F

    Returns
    -------
    jnp.ndarray
        flattened array of stage approximations evaluated in G
    """
    return jnp.ravel(Z) - dm * jnp.dot(A_d, jnp.ravel(F(Z,z0)))

def stop_criterion(dz_old:jnp.ndarray, dz_new:jnp.ndarray, k:int, eta_old:float, tol:float=1e-16, kappa:float=1e-4, u_round:float=1e-16):
    """stop_criterion checks if an acceptable convergence for the Newton iterations is achieved.

    The stop criterion is described in the Master's Thesis (pp. 52, step five).
    
    Parameters
    ----------
    dz_old : jnp.ndarray
        old solution to linear system in Newton iterations
    dz_new : np.ndarray
        new solution to linear system in Newton iterations
    k : int
        current Newton iteration number
    eta_old : float
        former approximate to geometric series (unused, see reference material)
    tol : float, optional
        accepted tolerance (can be related to discretization error instead of machine precision), by default 1e-14
    kappa : float, optional
        some constant between 1e-1 and 1e-2, by default 1e-4
    u_round : float, optional
        machine precision (unused), by default 1e-16

    Returns
    -------
    stop : boolean
        whether to stop Newton iterations or not
    eta : float
        current approximation to the geometric series associated with the improvements of the Newton iterations (unused)

    """
    try:
        # eta:float = max(eta_old, u_round) ** 0.8 # Doesn't really work as initial value...
        theta = jax.lax.cond(k==0, 
            lambda dy_new, dy_old: 0.5, 
            lambda dy_new, dy_old: jnp.linalg.norm(dy_new, ord=2)/jnp.linalg.norm(dy_old, ord=2), 
            dz_new, dz_old)
        
        eta:float = theta / (1 - theta)
        stop =  (eta * jnp.linalg.norm(dz_new, ord=2) <= kappa * tol) # Can improve tol by more precise evaluation of discretization error
    except ZeroDivisionError:
        stop = True
        eta = 0
    return stop, eta

def newton_cond(val):
    """Conditional function to stop Newton Iterations"""
    *_, k = val 
    return k < 10 

def newton_bod(val, G:function):
    """Body function of Newton Iterations - this is what happens inside newton_update's jax.lax.while_loop."""
    Y, dm, y0, dY, lu_piv, eta_old, k = val

    rhs = - G(Y, y0, dm)
    dY_old = jnp.ravel(dY) # Alt. (1 + jnp.int(not k)) 
    dY = jla.lu_solve(lu_piv, rhs)
    Y += jnp.reshape(dY, jnp.shape(Y))

    stop, eta_old = stop_criterion(dY_old, dY, k, eta_old) # Tol set in stop criterion function
    k = jax.lax.cond(stop, lambda k: 10, lambda k: k+1, k)

    return (Y, dm, y0, dY, lu_piv, eta_old, k)

def newton_update(z0:jnp.ndarray, Gamma:jnp.ndarray, dm:jnp.float64, d:int, stages:int, G:function):
    """
    newton_update performs Newton iterations to solve the nonlinear system of the IRK solver (procedure described in Thesis, pp. 52)

    Parameters
    ----------
    z0 : jnp.ndarray
        initial value of the Newton iterations (the previous timestep of the variable)
    Gamma : jnp.ndarray
        matrix returned by calc_gamma
    dm : jnp.float64
        current time increment of the modified measure dm = sigma_0 * dt + sigma * dW
    d : int
        number of spatial dimentions
    stages : int
        number of stages of the IRK method
    G : function
        see function definition of G

    Returns
    -------
    Z : jnp.ndarray
        Array of the accepted approximations from the stages of the RK method.
    """
    Z = jnp.zeros((stages, d))
    dZ = jnp.zeros(d * stages) 
    eta_old = jnp.float64(1e-16)
    lu_piv = jla.lu_factor(Gamma)
    k = 0
    val = (Z, dm, z0, dZ, lu_piv, eta_old, k)

    cond_fun = lambda val: newton_cond(val)
    body_fun = lambda val: newton_bod(val, G=G)

    Z, *_ = jax.lax.while_loop(cond_fun, body_fun, val)
    return Z


def one_step_method(x_old:jnp.ndarray, dm:jnp.float64, rk:RK1, calc_gamma:function, newt_upd:function, F:function):
    """
    one_step_method calculates one-step approximation from x_old using the IRK method given by rk.

    Parameters
    ----------
    x_old : jnp.ndarray
        integrated variable at previous timestep
    dm : jnp.float64
        current time increment of the modified measure dm = sigma_0 * dt + sigma * dW
    rk : RK1
        class with coefficients of the IRK method
    calc_gamma : function
        see function definition of calc_gamma
    newt_upd : function
        newton update routine used in this method
    F : function
        see function definition of F

    Returns
    -------
    x_new : jnp.ndarray
        the variable approximated in the next timestep
    """
    Gamma = calc_gamma(x_old, dm)
    Y = newt_upd(x_old, Gamma, dm)
    x_new = x_old + jnp.dot(rk.b, F(Y,  x_old)) * dm
    return x_new

def map_simulations(x_batch:jnp.ndarray, dm_batch:jnp.ndarray, one_step:function):
    """
    map_simulations maps short-hand version of one_step_method across one batch of simulations

    Parameters
    ----------
    x_batch : jnp.ndarray
        array of simulated variables in one batch, i.e. of shape (batch_simulations, d)
    dm_batch : jnp.ndarray
        array of simulated modified measures in one batch, i.e. of shape (batch_simulations,)
    one_step : function
        short-hand version of one_step_method (see definition of one_step_method and irk_sde_solver)

    Returns
    -------
    jnp.ndarray
        the next timestep of the simulations in the batch
    """
    return jax.vmap(one_step, in_axes=0)(x_batch, dm_batch)


def body_scan(x_old:jnp.ndarray, dM_n:jnp.ndarray, map_sims:function):
    x_new = map_sims(x_old, dM_n)
    return x_new, x_new

def scan_time(x:jnp.ndarray, dM:jnp.ndarray, body_scan:function):
    """
    scan_time integrates x numerically with one-step method (body_scan) w.r.t. time over batch simulations

    Equivalent to looping over range(batch_simulations) and range(timesteps).

    Parameters
    ----------
    x : jnp.ndarray
        array of dimension (timesteps + 1, batch_simulations, d) with first elements equal to x0
    dM : jnp.ndarray
        array of simulated increments of measure dm = sigma_0 * dt + sigma * dW with shape (timesteps, batch_simulations)
    body_scan : function
        function doin the work in the loops.

    Returns
    -------
    jnp.ndarray
        integrated variable
    """
    _, approximation = jax.lax.scan(body_scan, x[0], dM)
    x = x.at[1:].set(approximation)
    return x

def scan_batch_map(x:jnp.ndarray, dM:jnp.ndarray, scan_time:function, backend:str):
    """
    scan_batch_map maps scan_time in parallel over batches (second axis)

    Parameters
    ----------
    x : jnp.ndarray
        array of dimension (timesteps + 1, batches, batch_simulations, d) with first elements equal to x0
    dM : jnp.ndarray
        simulated increments of measure dm = sigma_0 * dt + sigma * dW with shape (timesteps, batches, batch_simulations)
    scan_time : function
        see definition of scan_time
    backend : str
        backend used for computations (cpu, gpu, tpu)

    Returns
    -------
    jnp.ndarray
        computes 
    """
    return jax.pmap(scan_time, in_axes=1, out_axes=1, backend=backend)(x, dM)


# jaxed irk_sde_solver
##### ---------------------------------------------------------------------------------------------------- ######

def irk_sde_solver(x0:np.ndarray, rk:RK, f:function, df:function, sigma_0:float, sigma:float, dW:np.ndarray,
        t0=0., T=1., backend='cpu'
    ):
    """irk_sde_solver solves a single-integrand SDE using supplied features and rk method using jax. 

    solves SDEs of the form

    dX(t) = f(X) * (sigma_0 * dt + sigma o dW(t)),  t in [t0, T], x(t0) = x0,
    
    with stepsize, batches and number of batch simulations inferred from the supplied simulated Wiener Process dW.

    Note: All the functions used internally are redefined with lambda functions to make them static, which allows for JIT-compilation.

    Parameters
    ----------
    x0 : np.ndarray
        initial value(s) of differential equation
    rk : RK
        IRK one-step method used to approximate the solution
    f : function
        vector field of the differential equation
    df : function
        gradient of vector field f.
    sigma_0 : float
        drift constant
    sigma : float
        diffusion constant
    dW : jnp.array
        simulated Wiener Process
    t0 : float, optional
        initial time, by default 0.
    T : float, optional
        terminal time, by default 1.
    backend : str, optional
        computer backend on which the computations are performed, by default 'cpu'

    Returns
    -------
    np.ndarray
        approximated variable x of shape (timesteps + 1, batches, batch_simulations, d)
    """
    # Convert to jax.numpy.float64 DeviceArray's
    sigma_0, sigma, t0, T, dW = jnp.float64(sigma_0), jnp.float64(sigma), jnp.float64(t0), jnp.float64(T), jnp.float64(dW)
    
    # Get some shapes and constants
    timesteps, batches, batch_simulations = dW.shape
    dt = jnp.float64((T-t0)/timesteps)
    d = x0.shape[0]

    x_shape = (timesteps+1, batches, batch_simulations, d)
    dM = sigma_0 * dt + sigma * jnp.float64(dW)

    # initialize x
    x = np.zeros(x_shape)
    x[0, ..., :] = x0
    
    stages = rk.b.shape[0] # rk.k differs for Lobatto and Gauss methods and doesn't work here

    rk = RK1(rk)
    I_d = jnp.eye(stages * d) 
    A_d = jnp.kron(rk.A, jnp.eye(d)) # Same dimension as I_m

    # Helper functions
    F_local = lambda Z, z0: F(Z, z0, f=f)
    calc_gamma_local = lambda y0, dm: calc_gamma(y0, dm, I_d=I_d, A=rk.A, df=df)
    G_local = lambda Z, z0, dm: G(Z, z0, dm, A_d=A_d, F=F_local)

    # Newton update short-hand
    newton_update_local = lambda z0, Gamma, dm: newton_update(z0, Gamma, dm, d=d, stages=stages, G=G_local)
    
    # One step of IRK method short-hand
    one_step_method_local = lambda x_old, dm: one_step_method(x_old, dm, rk=rk, calc_gamma=calc_gamma_local, newt_upd=newton_update_local, F=F_local)

    # mappping over simulations short-hand
    map_simulations_local = lambda x_n, dm_n: map_simulations(x_n, dm_n, one_step_method_local)

    # body function of scan short-hand
    body_scan_local = lambda x_old, dM_n: body_scan(x_old, dM_n, map_simulations_local)

    # scanning over time (in each batch)
    scan_time_local = lambda x_batch, dM_batch: scan_time(x_batch, dM_batch, body_scan_local)

    # mapping over batches
    x = scan_batch_map(x, dM, scan_time_local, backend)

    return x
##### -------------------------------------------------------------------------------------- #####


def stop_criterion_old(dz_old:np.ndarray, dz_new:np.ndarray, k:int, eta_old:float, tol:float=1e-14, kappa:float=1e-2, u_round:float=1e-16):
    """stop_criterion_old checks if an acceptable convergence for the Newton iterations is achieved

    Follows procedure from Hairer and Wanner's "Solving ODEs II" Ch.IV.8 pp. 120-121 to calculate a stop criterion using current and former `dy`,
    as well as a method specific tolerance level `tol` similar to the local discretization error of the method,
    roundoff error of the computer `u_round`, iteration number `k`, an old temporary value `eta_old` and some constant `kappa`.

    Parameters
    ----------
    dz_old : np.ndarray
        old solution to linear system in Newton iterations
    dz_new : np.ndarray
        new solution to linear system in Newton iterations
    k : int
        current iterate
    eta_old : float
        former approximate to geometric series (unused, see reference material)
    tol : float, optional
        accepted tolerance (can be related to discretization error instead of machine precision), by default 1e-14
    kappa : float, optional
        some constant between 1e-1 and 1e-2, by default 1e-2
    u_round : float, optional
        machine precision (unused), by default 1e-16

    Returns
    -------
    stop : boolean
        whether to stop Newton iterations or not
    eta : float
        current approximation to the geometric series associated with the improvements of the Newton iterations (unused)

    """
    
    try:
        if k == 0:
            theta = 0.5
            # eta:float = max(eta_old, u_round) ** 0.8 # Doesn't really work
        else:
            theta = la.norm(dz_new, ord=2)/la.norm(dz_old, ord=2)
    
        eta:float = theta / (1 - theta)

        if eta * la.norm(dz_new, ord=2) <= kappa * tol: # Can improve tolerance estimate by more precise evaluation of discretization error
            stop = True
        else:
            stop = False
    except ZeroDivisionError:
        stop = True
        eta = 0
    # print("Theta = {}, Eta = {}".format(theta, eta))
    # print("Tolerance:", kappa * tol)
    # print("Criterion:", eta * la.norm(dy_new))

    return stop, eta


def newton_update_old(z0:np.ndarray, G:function, Gamma:np.ndarray, stages:int=None, tol:float=1e-16):
    """
    newton_update_old performs Newton iterations to solve the nonlinear system of the IRK solver (procedure described in Thesis, pp. 52),
    but uses numpy and scipy functions rather than jax.

    Parameters
    ----------
    z0 : np.ndarray
        initial approximation to the stage updates (i.e. x_old)
    G : function
        calculates the right-hand side vector of the linear system 
    Gamma : np.ndarray
        constant matrix on thhe right-hand side of the linear system
    stages : int, optional
        number of stages of the applied irk method, by default None
    tol : float, optional
        convergence tolerance of iterations, by default 1e-16

    Returns
    -------
    np.ndarray
        computed stage approximations
    """
    # y0: (d,)-array
    # G: R^(k+1,d) -> R^m
    # Gamma: (m,m)-array
    # stages: k+1

    d = z0.shape[0]
    m = d * stages
    Z:np.ndarray = np.zeros((stages, d))
    dZ:np.ndarray = np.zeros(m)
    eta_old = 1e-16

    lu_piv = la.lu_factor(Gamma)

    
    for k in range(10):
        rhs = -G(Z)
        dZ_old:np.ndarray = dZ.flatten()
        dZ = la.lu_solve(lu_piv, rhs)
        Z += dZ.reshape(stages, d) # Careful about order!

        stop, eta_old = stop_criterion_old(dZ_old, dZ, k, eta_old, tol)
        if stop:
            break
    return Z

def irk_sde_solver_old(x0:np.ndarray, rk:RK, f:function, df:function, sigma_0:float, sigma:float, dW:np.ndarray,
        t0:float=0, T:float=1.,
    )->np.ndarray:
    """`irk_sde_solver_old` solves SDE problems of the form dX(t) = f(X) (dt + sigma * dW(t)) using old and slow implementation

    Old and inefficient implementation of irk_sde_solver using numpy and scipy which works on Windows.
    Not parallelized, nor JIT-compiling.

    Parameters
    ----------
    x0 : np.ndarray
        initial value of problem
    rk : RK_butcher
        Butcher table of method used by the solver 
    f : function
        Vector field function of problem
    df : function
        Jacobian of f
    sigma_0 : float, optional
        drift constant
    sigma : float
        diffusion constant
    dW : np.ndarray
        Simulated Wiener increments
    t0 : float, optional
        Initial time, by default 0
    T : float, optional
        Terminal time, by default 1.

    Returns
    -------
    np.ndarray
        The approximated process x of dimension (timesteps, batches, batch_simulations, len(x0))
    """
    
    if type(sigma) != np.ndarray: sigma = np.array([sigma])
    # Initialize some variables
    timesteps, batches, batch_simulations = dW.shape
    dt = (T-t0)/timesteps
    d = x0.shape[0] # x0 assumed (d,)- array
    x_shape = (timesteps+1, batches, batch_simulations, d)
    x = np.zeros(x_shape)
    x[0, ..., :] = x0

    if sigma == 0: # Solves deterministic problem
        x_shape = (timesteps+1, 1, 1, d)
        dM = np.full(x_shape[:-1], sigma_0 * dt)
    else: # Solves stochastic problem
        x_shape = (timesteps+1, batches, batch_simulations, d)
        dM = sigma_0 * dt + sigma * dW

    # Prepare for newton iterations
    k = rk.b.shape[0] # rk.k is different for Lobatto methods and doesn't work 
    m = k * d
    F = lambda Y, y0: np.array([f(y0 + y) for y in Y])

    for b in range(batches):
        for s in range(batch_simulations):
            for n in range(timesteps):

                z0 = x[n, b, s]
                dm = dM[n, b, s]

                Gamma = np.eye(m) - dm * np.kron(rk.A, df(z0))
                G = lambda Z: Z.ravel() - dm * np.kron(rk.A, np.eye(d)) @ F(Z, z0).ravel() # Mind the ravel order
                Z = newton_update_old(z0, G, Gamma, stages=k, print_iterations=True)
                x[n+1, b, s] = z0 + np.einsum(", i, ij -> j", dm, rk.b, F(Z, z0))

    return x


def irk_ode_solver(x0:jnp.array, rk:RK, f:function, df:function,
        t0=jnp.float64(0.), T=jnp.float64(1.), dt=jnp.float64(1/2**6), sigma_0=jnp.float64(1),
    ):
    """Solves ODE of the form `x'(t) = sigma_0 * f(x)` (sigma_0 by default 1) using an arbitrary IRK method
    
    Note: Equivalent to irk_sde_solver with sigma_0=1 and sigma=0; uses irk_sde_solver as subroutine."""
    timesteps = int((T-t0)/dt)
    dW_dummy = np.empty((timesteps, 1, 1)) # Only used to get timesteps from shape
    return irk_sde_solver(x0, rk, f, df, sigma_0=sigma_0, sigma=0, dW=dW_dummy, t0=t0, T=T)

# -------------------------------------------------------------------------- #
# Solver for SHS with additive noise and drift (in the end unused in Thesis)

def symmetric_newton_update(y0, Gamma, dt, d, G:function):
    """Newton Update variant used by the symmetric_solver method, which has stages=1 due to the structure of the methods."""
    return jnp.squeeze(newton_update(y0, Gamma, dm=dt, d=d, stages=1, G=G))

def symmetric_one_step(x_old, dm, calc_gamma, newt_upd):
    """Calculate one-step approximation of simulated variable with symmetric RK method"""
    Gamma = calc_gamma(x_old, dm)
    return x_old + newt_upd(x_old, Gamma, dm)

def get_method_system(dt, d, f, df, method="midpoint", theta=None):
    """
    get_method_system get correct function and linear system for the method specified in method

    Parameters
    ----------
    dt : float
        temporal step-length (step-length in t)
    d : int
        dimension of problem
    f : function
        expression for the differential equation to be solved
    df : function
        gradient of f
    method : str
        name of method applied, either "midpoint", "kahan", "mq" or "lobatto" (Lobatto-4 type IIIA)
    theta : float, optional
        value of parameter in theta method(mq or lobatto), by default 1.

    Returns
    -------
    G : function
        Function definition used in Newton iterations
    Gamma : np.ndarray
        Approximation to gradient of G
    p : int
        Order of numerical method

    """
    dt = jnp.float64(dt)
    if method == "midpoint": # is basic, but works well
        calc_gamma= lambda y0, dt:      jnp.eye(d) - dt/2 * df(y0)
        
        def G(y, y0, dt):
            y = jnp.squeeze(y) # NOTE: These must be here the way the Newton iterations are built!
            return y - f(y0 + y/2) * dt

        p = 2
    elif method == "kahan": # Conserves quadratic vector fields (in theory)
        calc_gamma =lambda y0, dt: jnp.eye(d) - dt/2 * df(y0)
        
        def G(y, y0, dt): 
            y = jnp.squeeze(y)
            return y - dt/2 * (-f(y0) + 4 * f(y0 + y/2) - f(y0 + y))

        p = 2
    else:
        if theta is not None: pass
        elif method == "mq":        theta = 0. # Conserves quartic vector fields - should be identical to RK_butcher(k=2, s=1, quadrature=0)
        elif method == "lobatto":   theta = 1. # lobatto order 4 - should be identical to irk_ode_solver using RK_butcher(s=2, quadrature=3).
        else: theta = 1.

        def calc_gamma(y0, dt, d=d, df=df):
            df0 = df(y0)
            return jnp.eye(d) - dt/2 * df0 + theta * dt **2 / 12 * jnp.dot(df0, df0)
        
        def G(y, y0, dt):
            y = jnp.squeeze(y)
            return y - dt/6 * (f(y0) + 4 * f(y0 + y/2 + theta * dt / 8 * (f(y0) - f(y0 + y))) + f(y0 + y))
        p = 4
    
    return jit(G), jit(calc_gamma), p


# Symmetric solver based on burrage and burrage, now jaxed.
def symmetric_solver(x0:np.array, f:function, df:function,
        t0=0., T=1., dt= 1 / 2**6, 
        method:str="midpoint", theta=None
    ):
    """symmetric_solver approximates a Hamiltonian ODE with one of four symmetric numerical schemes

    Generates a numerical approximation to the equation 
    :math:`x'(t) = f(x)`
    from initial values `x0`, vector field `f` and its gradient `df` using the scheme given in the `method` string.
    
    
    As all the solvers are implicit, an approximation to the next step is calculated using simplified Newton iterations.


    Parameters
    ----------
    x0 : np.ndarray
        initial value of variable x to be approximated
    f : function
        vector field function
    df : function
        gradient of f
    t0 : float, optional
        initial time value, by default 0.
    T : float, optional
        terminal time value, by default 40.
    dt : float, optional
        time-step, by default 1e-1
    method : str, optional
        name of method applied, by default "midpoint"
    theta : float, optional
        parameter of theta method, by default 0.

    Returns
    -------
    np.ndarray
        approximated value of x.
    """
    # Convert to jnp.float64
    t0, T, dt = jnp.float64(t0), jnp.float64(T), jnp.float64(dt)

    # 
    timesteps = int((T-t0)/dt)
    d = x0.shape[0]
    x = np.zeros((timesteps+1, d))
    x[0] = x0

    # Shorthand jitted functions
    G_local, calc_gamma, p = get_method_system(dt, d, f, df, method, theta=theta)
    newt_upd = jit(lambda x_old, Gamma, dt: symmetric_newton_update(x_old, Gamma, dt, d=d, G=G_local))
    one_step_method = jit(lambda x_old, dt: symmetric_one_step(x_old, dt, calc_gamma, newt_upd))

    # Calculate the solution
    for n in range(timesteps): x[n+1] = one_step_method(x[n], dt)
    
    return x

def drift_solver(
        x0:np.ndarray, f:function, df:function, sigma:np.ndarray, Z:np.ndarray,
        t0=0., T=1., 
        method:str="midpoint", theta=None, rk=None, backend='cpu'
    ):

    """
    drift_solver calculates approximation to an SHS (or indeed a general SDE) problem with additive noise of the form

    dX(t) = f(X) dt + sigma @ dW(t), X(0) = x0, f = J dH, t0 <= t <= T,

    using either a one-line expression for a named symmetric method or a supplied general rk method.

    Parameters
    ----------
    x0 : jnp.ndarray
        Initial value of simulated variable
    f : function
        Vector field function of differential equation
    df : function
        Jacoubian of of f
    sigma : jnp.ndarray
        Diffusion matrix in equation of shape (d, diffusion_terms), where the first d/2 rows are zero
    Z : jnp.ndarray
        Dimulated Wiener process of shape (timesteps, 2, batches, batch_simulations, diffusion_terms)
    t0 : float, optional
        Initial time, by default 0
    T : float, optional
        Terminal time, by default 1
    method : str, optional
        Name of numerical scheme, by default "midpoint"
    theta : float, optional
        Value of theta in "theta" scheme (either 0 or 1), by default 1
    rk : RK_butcher-like, optional
        Class with attributes A, b, c corresponding to coefficients of an IRK method
    backend : str, optional
        Machine back-end on which the computations are performed ('cpu', 'gpu' or 'tpu')
    Returns
    -------
    x : np.ndarray
        Array of approximated values with shape (timesteps + 1, batches, batch_simulations, d)
        
    """
    # Convert parameters to jnp.float64 immutables
    t0, T, sigma, Z = jnp.float64(t0), jnp.float64(T), jnp.float64(sigma), jnp.float64(Z)
    # Note: Sigma is expected to be of shape (d, diffusion_terms)
    timesteps, _, batches, batch_simulations, diffusion_terms = Z.shape # _ = 2 by method definition
    dt = (T - t0) / timesteps
    d = x0.shape[0]
    try: assert np.allclose(sigma.shape, (d, diffusion_terms))
    except AssertionError:
        return print("sigma is wrongly initialized.")
    
    # Initialize approximated variable
    x:np.ndarray = np.zeros((timesteps+1, batches, batch_simulations, d))
    x[0, ..., :] = x0

    # Shorthand functions, which are jitted 

    if rk is None: # Calculate using one of the explicit expressions for the symmetric methods
        G_local, calc_gamma_local, p = get_method_system(dt, d, f, df, method, theta=theta)
        newt_upd = jit(lambda x_old, Gamma, dt: symmetric_newton_update(x_old, Gamma, dt, d=d, G=G_local))
        one_step_method_local = jit(lambda x_old, dt: symmetric_one_step(x_old, dt, calc_gamma_local, newt_upd))
    else: # Calculate using an arbitrary IRK method as given by the class instance rk
        rk = RK1(rk)
        stages = rk.b.shape[0]
        I_d = jnp.eye(stages * d) 
        A_d = jnp.kron(rk.A, jnp.eye(d)) # Same dimension as I_d
        F_local = jit(lambda Y, y0: F(Y, y0, f=f))
        calc_gamma_local = jit(lambda y0, dm: calc_gamma(y0, dm, I_d=I_d, A=rk.A, df=df))
        G_local = jit(lambda Y, y0, dm: G(Y, y0, dm, A_d=A_d, F=F_local))
        newton_update_local = jit(lambda y0, Gamma, dm: newton_update(y0, Gamma, dm, d=d, stages=stages, G=G_local))
        one_step_method_local = jit(lambda x_old, dm: one_step_method(x_old, dm, rk=rk, calc_gamma=calc_gamma_local, newt_upd=newton_update_local, F=F_local))

    map_simulations_local = lambda x_batch, dt: jax.vmap(one_step_method_local, in_axes=(0, None))(x_batch, dt)
    map_batches_local = lambda x_old, dt: jax.pmap(map_simulations_local, in_axes=(0, None), devices=jax.devices(), backend=backend)(x_old, dt)
    
    sigma = jnp.transpose(sigma) # to ensure correct indices in the calculation below
    for n in range(timesteps):
            x_old = x[n] + jnp.dot(Z[n,0], sigma)
            x_updated = map_batches_local(x_old, dt)
            x[n+1] = x_updated + jnp.dot(Z[n,1], sigma)

    return x


if __name__ == "__main__":
    pass
