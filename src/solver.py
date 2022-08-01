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
        batches=10, batch_simulations=100, seed=100,
        diffusion_terms=0, k=2, truncate=True
    ):
    """truncated_wiener Simulate Wiener Process with truncations to ensure solvability

    Generates array of truncated variables sampled from a normal distribution with each element a ~ N(0, dt).
    The dimension of the array is (timesteps, batches, batch_simulations, diffusion_terms), where
    
        timesteps = (T - t0) / dt
    
    and the rest are keyword arguments. The truncation bound is given by formula

        A_h = sqrt(2 * k * |log(dt)|)
    
    and the random number generator used is PCG64 (default in numpy as of 2022).
    
    The truncation is based on Stochastic Numerics in Mathematical Physics by Milstein & Tretyakov.

    Parameters
    ----------
    t0 : int, optional
        start time, by default 0
    T : int, optional
        stop time, by default 1
    dt : float, optional
        stepsize, by default 1e-1
    batches : int, optional
        Number of simulated batches, by default 10
    batch_simulations : int, optional
        Number of simulated Wiener Processes in each batch, by default 100
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
        simulated Wiener increments, possibly truncated
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
    """Truncate supplied Wiener process using jax functions"""
    A_h = jnp.sqrt(2*k*jnp.abs(jnp.log(dt)))
    # Truncating variable
    return jnp.where(jnp.abs(dW) < A_h, dW, jnp.sign(dW) * A_h)

def generate_wiener_jax(dt, seed, timesteps, batches, batch_simulations, diffusion_terms=0):
    """Generate wiener process using jax functions."""
    noise_shape = [timesteps, batches, batch_simulations]
    if diffusion_terms: noise_shape.append(diffusion_terms)
    key = jax.random.PRNGKey(seed)
    return jnp.sqrt(dt) * jax.random.normal(key, shape=noise_shape)


def truncated_wiener_jax(
        t0=jnp.float64(0), T=jnp.float64(1), dt=jnp.float64(1e-1), 
        batches=20, batch_simulations=1000, seed=100, 
        diffusion_terms=0, k=2, truncate=True):
    """Jax version of truncated_wiener anologous to truncated_wiener."""
    
    timesteps = int((T-t0)/dt)
    dW = generate_wiener_jax(dt, seed, timesteps, batches, batch_simulations, diffusion_terms=diffusion_terms)
    
    if truncate: return truncate_wiener(dW, k, dt)
    else: return dW

# Support sizes for the jaxed irk_sde_sovler

class RK1():
        """Temporary class which stores A, b, and c from RK_butcher as jnp.ndarrays (param class only allows for np.array for standard parameters)."""
        def __init__(self, rk:RK):
            self.A = jnp.float64(rk.A)
            self.b = jnp.float64(rk.b)
            self.c = jnp.float64(rk.c)

def F(Y, y0, f:callable):
    """f applied to the approximations of x_new, i.e. y0 + y, for all stages of the IRK method"""
    return jax.vmap(lambda y: f(y0 + y))(Y)
    
def calc_gamma(y0, dm, I_d, A, df:callable):
    "Left-hand side of the Newton iterations"
    return I_d - dm * jnp.kron(A, df(y0))

def G(Y, y0, dm, A_d, F:callable):
    "Function on the right-hand side of the Newton iterations"
    return jnp.ravel(Y) - dm * jnp.dot(A_d, jnp.ravel(F(Y,y0)))

def stop_criterion(dy_old:jnp.ndarray, dy_new:jnp.ndarray, k:int, eta_old:float, tol:float=1e-16, kappa:float=1e-4, u_round:float=1e-16):
    """stop_criterion checks if an acceptable convergence for the Newton iterations is achieved

    Follows procedure from Hairer and Wanner's "Solving ODEs II" Ch.IV.8 pp. 120-121 to calculate a stop criterion using current and former `dy`,
    as well as a method specific tolerance level `tol` similar to the local discretization error of the method,
    roundoff error of the computer `u_round`, iteration number `k`, an old temporary value `eta_old` and some constant `kappa`.


    Parameters
    ----------
    dy_old : np.ndarray
        old solution to linear system in Newton iterations
    dy_new : np.ndarray
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
        # eta:float = max(eta_old, u_round) ** 0.8 # Doesn't really work as initial value...
        theta = jax.lax.cond(k==0, 
            lambda dy_new, dy_old: 0.5, 
            lambda dy_new, dy_old: jnp.linalg.norm(dy_new, ord=2)/jnp.linalg.norm(dy_old, ord=2), 
            dy_new, dy_old)
        
        eta:float = theta / (1 - theta)
        stop =  (eta * jnp.linalg.norm(dy_new, ord=2) <= kappa * tol) # Can improve tol by more precise evaluation of discretization error
    except ZeroDivisionError:
        stop = True
        eta = 0
    return stop, eta

def newton_cond(val):
    """Conditional function to stop Newton Iterations"""
    *_, k = val 
    return k < 10 

def newton_bod(val, G:callable):
    """Body function of Newton Iterations - this is what happens inside the Newton Iterations' while_loop."""
    Y, dm, y0, dY, lu_piv, eta_old, k = val

    rhs = - G(Y, y0, dm)
    dY_old = jnp.ravel(dY) # Alt. (1 + jnp.int(not k)) 
    dY = jla.lu_solve(lu_piv, rhs)
    Y += jnp.reshape(dY, jnp.shape(Y))

    stop, eta_old = stop_criterion(dY_old, dY, k, eta_old) # Tol set in stop criterion function
    k = jax.lax.cond(stop, lambda k: 10, lambda k: k+1, k) #TODO: make jnp.nan and mask if reaching 10 iterations!

    return (Y, dm, y0, dY, lu_piv, eta_old, k)

def newton_update(y0, Gamma, dm, d, stages, G:callable):
    """Performs Newton Iterations to solve a nonlinear system connected to the IRK solver"""
    Y = jnp.zeros((stages, d))
    dY = jnp.zeros(d * stages) 
    eta_old = jnp.float64(1e-16)
    lu_piv = jla.lu_factor(Gamma)
    k = 0
    val = (Y, dm, y0, dY, lu_piv, eta_old, k)

    cond_fun = lambda val: newton_cond(val)
    body_fun = lambda val: newton_bod(val, G=G)

    # Loopeti-loop
    Y, *_ = jax.lax.while_loop(cond_fun, body_fun, val)
    return Y


def one_step_method(x_old, dm, rk, calc_gamma, newt_upd, F):
    "Calculates one-step approximation from x_old with step dm using the IRK method given by rk."
    Gamma = calc_gamma(x_old, dm)
    Y = newt_upd(x_old, Gamma, dm)
    x_new = x_old + jnp.dot(rk.b, F(Y,  x_old)) * dm
    return x_new

def map_simulations(x_batch, dm_batch, one_step:callable):
        "Maps one_step across batch_simulations in x_batch and dm_batch."
        return jax.vmap(one_step, in_axes=0)(x_batch, dm_batch)


def body_scan(x_old, dM_n, map_sims):
    x_new = map_sims(x_old, dM_n)
    return x_new, x_new

def scan_time(x:jnp.float64, dM, body_scan):
    _, approximation = jax.lax.scan(body_scan, x[0], dM)
    x = x.at[1:].set(approximation)
    return x

def scan_batch_map(x, dM, scan_time, backend):
    return jax.pmap(scan_time, in_axes=1, out_axes=1, backend=backend)(x, dM)


# jaxed irk_sde_solver
##### ---------------------------------------------------------------------------------------------------- ######

def irk_sde_solver(x0:np.array, rk:RK, f:callable, df:callable, sigma_0:float, sigma:float, dW:np.array,
        t0=0., T=1., backend='cpu'
    ):
    """irk_sde_solver solves a Single-Integrand SDE using supplied features and rk method using jax. 

    solves SDEs of the form

    dX(t) = f(X) * (sigma_0 * dt + sigma o dW(t)),  t in [t0, T], x(t0) = x0,
    
    with stepsize, batches and number of batch simulations inferred from the supplied simulated Wiener Process dW.

    Parameters
    ----------
    x0 : jnp.array
        initial value of differential equation
    rk : RK_butcher
        Implicit Runge-Kutta one-step method used to approximate the solution. 
    f : callable
        Vector field of the ODE.
    df : callable
        Gradient of vector field f.
    sigma_0 : float
        Drift constant
    sigma : float
        Diffusion constant
    dW : jnp.array
        Simulated Wiener Process
    t0 : float, optional
        Initial time, by default 0.
    T : float, optional
        Terminal time, by default 1.
    backend : str, optional
        Computer backend on which the accelerated linear algebra is performed, by default 'cpu'
    scan : bool, optional
        Whether to replace numpy iterations with jax.lax.scan (might be faster, but can cause issues combined with pmap)

    Returns
    -------
    np.array
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
    
    stages = rk.b.shape[0] # rk.k is different for Lobatto methods and doesn't work here

    rk = RK1(rk)
    I_d = jnp.eye(stages * d) 
    A_d = jnp.kron(rk.A, jnp.eye(d)) # Same dimension as I_m

    # Helper functions
    F_local = lambda Y, y0: F(Y, y0, f=f)
    calc_gamma_local = lambda y0, dm: calc_gamma(y0, dm, I_d=I_d, A=rk.A, df=df)
    G_local = lambda Y, y0, dm: G(Y, y0, dm, A_d=A_d, F=F_local)

    # Newton update
    newton_update_local = lambda y0, Gamma, dm: newton_update(y0, Gamma, dm, d=d, stages=stages, G=G_local)
    
    # One step of IRK method
    one_step_method_local = lambda x_old, dm: one_step_method(x_old, dm, rk=rk, calc_gamma=calc_gamma_local, newt_upd=newton_update_local, F=F_local)

    # mappping over simulations
    map_simulations_local = lambda x_n, dm_n: map_simulations(x_n, dm_n, one_step_method_local)

    # body function of scan
    body_scan_local = lambda x_old, dM_n: body_scan(x_old, dM_n, map_simulations_local)

    # scanning over time (in each batch)
    scan_time_local = lambda x_batch, dM_batch: scan_time(x_batch, dM_batch, body_scan_local)

    # mapping over batches
    x = scan_batch_map(x, dM, scan_time_local, backend)

    return x
##### -------------------------------------------------------------------------------------- #####


def stop_criterion_old(dy_old:np.ndarray, dy_new:np.ndarray, k:int, eta_old:float, tol:float=1e-14, kappa:float=1e-2, u_round:float=1e-16):
    """stop_criterion checks if an acceptable convergence for the Newton iterations is achieved

    Follows procedure from Hairer and Wanner's "Solving ODEs II" Ch.IV.8 pp. 120-121 to calculate a stop criterion using current and former `dy`,
    as well as a method specific tolerance level `tol` similar to the local discretization error of the method,
    roundoff error of the computer `u_round`, iteration number `k`, an old temporary value `eta_old` and some constant `kappa`.


    Parameters
    ----------
    dy_old : np.ndarray
        old solution to linear system in Newton iterations
    dy_new : np.ndarray
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
            theta = la.norm(dy_new, ord=2)/la.norm(dy_old, ord=2)
    
        eta:float = theta / (1 - theta)

        if eta * la.norm(dy_new, ord=2) <= kappa * tol: # Can improve tolerance estimate by more precise evaluation of discretization error
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


def newton_update_old(y0:np.ndarray, G:callable, Gamma:np.ndarray, stages=None, tol:float=1e-16, print_iterations=True):
    # y0: (d,)-array
    # G: R^(k+1,d) -> R^m
    # Gamma: (m,m)-array
    # stages: k+1

    d = y0.shape[0]
    m = d * stages
    Y:np.ndarray = np.zeros((stages, d))
    dY:np.ndarray = np.zeros(m)
    eta_old = 1e-16

    lu_piv = la.lu_factor(Gamma)

    
    for k in range(10):
        rhs = -G(Y)
        dY_old:np.ndarray = dY.flatten()
        dY = la.lu_solve(lu_piv, rhs)
        Y += dY.reshape(stages, d) # Careful about order!

        stop, eta_old = stop_criterion_old(dY_old, dY, k, eta_old, tol)
        if stop:
            break
    return Y

def irk_sde_solver_old(x0:np.ndarray, rk:RK, f:callable, df:callable, sigma_0:float, sigma:float, dW:np.ndarray,
        t0:float=0, T:float=1.,
    )->np.ndarray:
    """`irk_sde_solver_old` solves SDE problems of the form dX(t) = f(X) (dt + sigma * dW(t)) using old and slow implementation

    Old and inefficient implementation of irk_sde_solver which works on windows.
    Not parallelized, nor jitted.

    Parameters
    ----------
    x0 : np.ndarray
        initial value of problem
    rk : RK_butcher
        Butcher table of method used by the solver 
    f : callable
        Vector field function of problem
    df : callable
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

                y0 = x[n, b, s]
                dm = dM[n, b, s]

                Gamma = np.eye(m) - dm * np.kron(rk.A, df(y0))
                G = lambda Y: Y.ravel() - dm * np.kron(rk.A, np.eye(d)) @ F(Y, y0).ravel() # Mind the ravel order
                Y = newton_update_old(y0, G, Gamma, stages=k, print_iterations=True)
                x[n+1, b, s] = y0 + np.einsum(", i, ij -> j", dm, rk.b, F(Y, y0))

    return x



# Jax implementation of irk_solver for ODEs using the function above
def irk_ode_solver(x0:jnp.array, rk:RK, f:callable, df:callable,
        t0=jnp.float64(0.), T=jnp.float64(1.), dt=jnp.float64(1/2**6), sigma_0=jnp.float64(1),
    ):
    """Solves ODE of the form `x'(t) = sigma_0 * f(x)` (sigma_0 by default 1) using an arbitrary irk method
    
    Note: Uses irk_sde_solver as subroutine."""
    timesteps = int((T-t0)/dt)
    dW_dummy = np.empty((timesteps, 1, 1)) # Only used to get timesteps from shape
    return irk_sde_solver(x0, rk, f, df, sigma_0=sigma_0, sigma=0, dW=dW_dummy, t0=t0, T=T)

# Support functions for the symmetric solver
def symmetric_newton_update(y0, Gamma, dt, d, G:callable):
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
    f : callable
        expression for the differential equation to be solved
    df : callable
        gradient of f
    method : str
        name of method applied, either "midpoint", "kahan", "mq" or "lobatto" (Lobatto-4 type IIIA)
    theta : float, optional
        value of parameter in theta method(mq or lobatto), by default 1.

    Returns
    -------
    G : callable
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
def symmetric_solver(x0:np.array, f:callable, df:callable,
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
    f : callable
        vector field function
    df : callable
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
        x0:np.ndarray, f:callable, df:callable, sigma:np.ndarray, Z:np.ndarray,
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
    f : callable
        Vector field function of differential equation
    df : callable
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
