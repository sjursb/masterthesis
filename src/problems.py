from functools import partial

from local_variables import xla_flag



import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from problem_classes import SingleIntegrand

#### JAXXED CODE #####

jax.config.update("jax_enable_x64", True) # Needed for use of 64-bit numbers!

x0_jax = jnp.array([jnp.sqrt(3), 1., 1., 1.,])
@partial(jit, static_argnums=0)
def J(d): return jnp.kron(jnp.array([[0,1], [-1, 0]]), jnp.eye(d))

@jit
def hh_H_jax(x:np.ndarray, alpha=1/16):
    return 1/2 * jnp.linalg.norm(x, ord=2, axis=-1) ** 2 + alpha * (x[..., 0] * x[..., 1]**2 - 1/3 * x[..., 0]**3)

@partial(jit, static_argnames='d')
def hh_f_jax(x:np.ndarray, alpha:float=1/16, d=2):
    return jnp.dot(J(d), x + jnp.kron(jnp.array([alpha, 0]), jnp.array([x[1]**2 - x[0]**2, 2 * x[0] * x[1]])))

@partial(jit, static_argnames='d')
def hh_df_jax(x:np.ndarray, alpha:float=1/16, d=2):
    return jnp.dot(J(d), jnp.eye(2*d) + jnp.kron(2 * alpha * jnp.array([[1, 0], [0, 0]]), jnp.array([[-x[0], x[1]], [x[1], x[0]]])))

henon_heiles = SingleIntegrand(
    title = "Hénon-Heiles",
    x0 = np.array(x0_jax),
    hamiltonian = hh_H_jax,
    f = hh_f_jax,
    df = hh_df_jax,
    nu = 3,
    dt_max = 1 / 2 ** 4
)

J_old = lambda d: np.kron(np.array([[0,1],[-1,0]]), np.eye(d))

henon_heiles_old = SingleIntegrand(
    title = "Hénon-Heiles old",
    x0 = np.array(x0_jax),
    hamiltonian = lambda x: 1/2 * np.linalg.norm(x, ord=2, axis=-1) ** 2 + 1/16 * (x[..., 0] * x[..., 1]**2 - 1/3 * x[..., 0]**3),
    f = lambda x: np.dot(J_old(2), x + np.kron(np.array([1/16, 0]), np.array([x[1]**2 - x[0]**2, 2 * x[0] * x[1]]))),
    df = lambda x: np.dot(J_old(2), np.eye(4) + np.kron(2 * 1/16 * np.array([[1, 0], [0, 0]]), np.array([[-x[0], x[1]], [x[1], x[0]]]))),
    nu = 3,
    dt_max = 1 / 2 ** 4
)

def henon_heiles_H(x:np.ndarray, alpha=1/16):
    return 1/2 * jnp.sum(x**2, axis=0) + alpha * (x[0]*x[1]**2 - 1/3 * x[0]**3)

henon_heiles_autodiff = SingleIntegrand(
    title = "Hénon-Heiles autograd",
    x0 = np.array(x0_jax),
    hamiltonian = hh_H_jax,
    nu=3,
    dt_max = 1 / 2 ** 4
)

henon_heiles_autodiff.get_ode_from_H(henon_heiles_H)
## Double-Well problem with 1 dimension - From Burrage, Burrage (2014)
# Used in BB14/ApGV21/CCAL20(LR04)

def double_well_f(x, d=1):
    return jnp.dot(J(d), jnp.array([x[0]**3 - x[0], x[1]]))

def double_well_df(x, d=1):
    return jnp.dot(J(d), jnp.array([[3*x[0]**2 - 1, 0], [0, 1]]))

@jit
def double_well_hamiltonian(x, d=1):
    return 0.25 * x[..., 0] ** 4 - 0.5 * x[..., 0]**2 + 0.5 * x[..., 1]**2

double_well = SingleIntegrand(
    title       = "Double-Well",
    x0          = np.full(2, np.sqrt(2)),
    f           = jit(lambda x: double_well_f(x)),
    df          = jit(lambda x: double_well_df(x)),
    hamiltonian = jit(lambda x: double_well_hamiltonian(x)),
    nu          = 4,
)

def double_well_H(x): return 0.25 * x[0] ** 4 - 0.5 * x[0]**2 + 0.5 * x[1]**2

double_well_autodiff = SingleIntegrand(
    title       = "Double-Well",
    x0          = np.full(2, np.sqrt(2)),
    hamiltonian = double_well_hamiltonian,
    nu          = 4,
)
double_well_autodiff.get_ode_from_H(double_well_H)

## Irreversible polynomial Hamiltonian of degree 6 studied in Faou, Hairer (04), HLW(?) and BIT(2014)
@jit
def h6_H(x):
    q, p = x[..., 0], x[..., 1]
    return p**3/3 - p/2 + q**6/30 + q**4/4 - q**3/3 + 1/6
@jit
def irreversile6_f(x):
    q, p = x[0], x[1]
    return jnp.array([p**2 - .5, -(q**5/5 + q**3 - q**2)])
@jit
def h6_df(x):
    q, p = x[0], x[1]
    return jnp.array([[0, 2*p], [-(q**4 + 3*q**2 - 2*q), 0]])

h6 = SingleIntegrand(
    title       = r"H order 6",
    x0          = np.array([0,1]),
    f           = irreversile6_f,
    df          = h6_df,
    hamiltonian = h6_H,
    nu          = 6,
    dt_max      = 1 / 2 ** 4
)

def h6_H(x): return x[1]**3/3 -x[1]/2 + x[0]**6/30 + x[0]**4/4 - x[0]**3/3 + 1/6

h6_autodiff = SingleIntegrand(
    title       = r"H order 6",
    x0          = np.array([0,1]),
    hamiltonian = h6_H,
    nu          = 6,
    dt_max      = 1 / 2 ** 4
)

h6_autodiff.get_ode_from_H(h6_H)

# Rigid body as in DK16/HXW14/HLW
# Parameters in DK16: i1, i2, i3 = 2, 1, 2/3; x0 = [cos(1.1), 0, sin(1.1)]
# Parameters in HXW14: i1, i2, i3 = .8, .6, .2 - these are used as default params!
@jit
def rigid_body_f(x, i1=.8, i2=.6, i3=.2):
    S = jnp.array(
        [
            [0,   x[2]/i3,    -x[1]/i2],
            [-x[2]/i3,  0,    x[0]/i1 ],
            [x[1]/i2,   -x[0]/i1,    0]
        ]
    )
    return jnp.dot(S, x)

def rigid_body_df(x, i1=.8, i2=.6, i3=.2):

    a = (1/i3 - 1/i2)
    b = (1/i1 - 1/i3)
    c = (1/i2 - 1/i1)

    df = jnp.array([ # Possibly wrong - might need transposing
        [0,    a*x[2], a*x[1]],
        [b*x[2], 0,    b*x[0]],
        [c*x[1], c*x[0],    0]])

    return df

def rigid_body_hamiltonian(x, i1=.8, i2=.6, i3=.2):
    I = jnp.array([i1, i2, i3])
    return .5 * jnp.dot(x**2, 1 / I)

def rigid_body_casimir(x, i1=.8, i2=.6, i3=.2):
    return jnp.linalg.norm(x, ord=2, axis=-1)**2

rigid_body = SingleIntegrand(
    title = "Rigid Body Problem",
    x0 = np.array([np.cos(1.1), 0, np.sin(1.1)]),
    f = rigid_body_f,
    df = rigid_body_df,
    hamiltonian = rigid_body_hamiltonian,
    invariant = rigid_body_casimir,
    sigma = .5,
    nu = None, # Non-canonical problem!
)

# Hamiltonian QI with d=1 and exact solution
# Used in DK16, HXW14

def kubo_exact(t, W=None, sigma_0=1, sigma=1):
    # if W is None: return np.array([np.sin(t), np.cos(t)]).swapaxes(0,-1)
    # else:
    # t = np.einsum("i, i... -> i...", t, np.ones_like(W))
    W = W.squeeze()
    return np.array([np.sin(sigma_0 * t + sigma * W), np.cos(sigma_0 * t + sigma * W)]).swapaxes(0,-1)

harmonic_oscillator = SingleIntegrand(
    title       = "Harmonic Oscillator",
    x0          = np.array([0, 1]),
    f           = jit(lambda x: J(1) @ x),
    df          = jit(lambda x: J(1)),
    hamiltonian = jit(lambda x: 0.5 * jnp.linalg.norm(x, ord=2, axis=-1)**2),
    exact       = kubo_exact,
    nu          = 2,
    sigma       = 0.,
)

kubo_oscillator = SingleIntegrand(
    title       = "Kubo Oscillator",
    x0          = np.array([0, 1]),
    f           = jit(lambda x: J(1) @ x),
    df          = jit(lambda x: J(1)),
    hamiltonian = jit(lambda x: 0.5 * jnp.linalg.norm(x, ord=2, axis=-1)**2),
    exact       = kubo_exact,
    nu          = 2,
)

# Kepler problem
@partial(jit, static_argnums=1)
def kepler_hamiltonian(x, d=2):
    q, p = x[..., :d], x[..., d:]
    return .5 * jnp.linalg.norm(p, ord=2, axis=-1) ** 2 - 1 / jnp.linalg.norm(q, ord=2, axis=-1)


def kepler_H(x):
    d = x.shape[0]//2
    return .5 * jnp.sum(x[d:]) ** 2 - 1 / jnp.linalg.norm(x[:d], ord=2)

# kepler_hamiltonian = jax.vmap(kepler_H, in_axes=-1)

kepler_H_grad = jax.grad(kepler_H)

def kepler_f_autodiff(x, d=2):
    return jnp.dot(J(d), kepler_H_grad(x))

kepler_H_hessian = jax.hessian(kepler_H)
def kepler_df_autodiff(x, d=2):
    return jnp.dot(J(d), kepler_H_hessian(x))

@partial(jit, static_argnums=1)
def kepler_f(x, d=2):
    q, p = x[:d], x[d:]
    denominator = jnp.linalg.norm(q, ord=2)**3
    dHdq = jax.vmap(lambda q_i: q_i/denominator)(q)
    dHdp = p
    dH = jnp.array([*dHdq, *dHdp])
    return jnp.dot(J(d), dH)

kronecker_delta = jax.jit(lambda i, j: jnp.array(i==j, int))

@partial(jit, static_argnums=1)
def kepler_df(x, d=2):
    q, p = x[:d], x[d:]
    o = jnp.zeros((d,d))
    norm_q = jnp.linalg.norm(q, ord=2, axis=0)
    ddHdq = jax.vmap(lambda i, q_i: jax.vmap(lambda j, q_j: kronecker_delta(i, j)/norm_q**3 - 3*q_i * q_j / norm_q**5)(jnp.arange(d), q))(jnp.arange(d), q)
    ddHdp = jnp.eye(d)
    ddH = jnp.block([[ddHdq, o], [o, ddHdp]])
    return jnp.dot(J(d), ddH)

@partial(jit, static_argnums=1)
def kepler_angular_momentum(x, d=3):
    q, p = x[..., :d], x[..., d:]
    return jnp.cross(q, p, axis=-1)

def kepler_orbit_iv(e=0.5, verbose=False):
    """Generate kepler IVs from eccentricity parameter e based on formulas of HLW ch.I.2"""
    try: 
        x0 = np.array([1-e, 0, 0, np.sqrt((1+e)/(1-e))])
        if verbose: print(f"Kepler x0={x0}")
    except ZeroDivisionError:
        print(f"Eccentricity has to be between 0 <= |e| < 1 for orbital movement! Given value e={e}")
    return x0

kepler_2d = SingleIntegrand(
    title = "Kepler Problem",
    x0 = kepler_orbit_iv(),
    f = lambda x: kepler_f(x, d=2),
    df = lambda x: kepler_df(x, d=2),
    hamiltonian = lambda x: kepler_hamiltonian(x, d=2),
    invariant = lambda x: kepler_angular_momentum(x, d=2),
    nu = None,
    dt_max = 1 / 2 ** 7
)

kepler_2d_autodiff = SingleIntegrand(
    title = "Kepler Problem",
    x0 = kepler_orbit_iv(),
    hamiltonian = lambda x: kepler_hamiltonian(x, d=2),
    invariant = lambda x: kepler_angular_momentum(x, d=2),
    f = kepler_f_autodiff,
    df= kepler_df_autodiff,
    nu = None,
    dt_max = 1 / 2 ** 7
)


# kepler_3d = SingleIntegrand(
#     title = "3D Kepler Problem",
#     f = kepler_f,
#     df= kepler_df,
#     hamiltonian = kepler_hamiltonian,
#     nu = None,
#     # invariant = lambda x: kepler_angular_momentum(x)[..., 2], # Only the third angular momentum is considered, however all are preserved.
# )

#### NOT YET JAXXED CODE ####
# FIXME: Convert the problems beneath to jax code!!

fermi_pasta_ulam = SingleIntegrand()
 ## Biot-Savart potential, not polynomial, d=6 (BIT2014)

biot_savart_x0 = np.array([.5, 10., 0., -.1, -.3, .0])

def biot_savart_hamiltonian(x, alpha=-1., m=1.):
    d = x.shape[-1]//2
    rho = jnp.linalg.norm(x, ord=2, axis=0) # Square of rho

    return 1./(2.* m) * (
            (x[..., d+0] - alpha*x[..., 0]/rho**2 )**2 + 
            (x[..., d+1] - alpha*x[..., 1]/rho**2 )**2 + 
            (x[..., d+2] + alpha*np.log(np.sqrt(rho**2)))**2
        )

def biot_savart_H(x, alpha=-1, m=1.):
    d = x.shape[0]//2
    rho = jnp.linalg.norm(x, ord=2, axis=0)
    return 1./(2.*m) * ((x[d] - alpha * x[0]/rho**2)**2 + (x[d+1] - alpha * x[1]/rho**2) + (x[d+2] + alpha * jnp.log(rho)**2)**2)


biot_savart_particle = SingleIntegrand(
    x0          = biot_savart_x0,
    f           = jit(jax.grad(biot_savart_H)),
    df          = jit(jax.grad(biot_savart_H)),
    hamiltonian = biot_savart_hamiltonian,
)


# Kværnø, Debrabant problems:
# Non-hamiltonian QI with d=1 and exact solution
def root_exact(t, W=None, sigma_0=1., sigma=1.):
    if W is None: return np.sinh(t)
    else:
        t = np.einsum("i, i... -> i...", t, np.ones_like(W))
        return np.sinh(sigma_0 * t + sigma * W) # This works because the dimension is 1

root_problem = SingleIntegrand(
    title       = R"$f(x) = \sqrt{{1 + x^2}}$",
    x0          = np.zeros(1),
    f           = lambda x: np.sqrt(1 + x ** 2),
    df          = lambda x: np.diag(x / np.sqrt(1 + x**2)),
    exact       = root_exact,
    nu          = None
)

# Molecylar dynamics: 
# N-body model:   H = 1/2 * p^T M^(-1) p + epsilon_{ij} V(||q_i - q_j||), M = diag(m)

# Lennard-Jones oscillator from LR04 pp.4-5,18-20, 40-41,71 - non-polynomial! (Also HLW pp.19)
# One-body problem: H = 1/2 * ||p||^2  + V(q),    V(q) = epsilon * ((r/q)^12 - (r/q)^6)

def  lennard_jones_hamiltonian(x, r=1, epsilon=.25, d=1):
    q, p = x[..., 0], x[..., 1]
    rq = r/q
    V = epsilon * (rq ** 12 -  rq ** 6)
    return p**2 / 2 + V

def lennard_jones_f(x, r=1, epsilon=.25, d=1):
    q, p = x[0], x[1]
    rq = r/q
    dV = 6 * epsilon / q * (rq ** 6 - 2 * rq ** 12)
    return np.einsum("ij, j", J(d), [dV, p])

def lennard_jones_df(x, r=1, epsilon=.25, d=1):
    q, p = x[0], x[1]
    rq = r/q
    ddV = 6 * epsilon / q**2 * (26 * rq**12 - 7 * rq ** 6)
    return np.einsum("ij, jk", J(1), [[ddV, 0], [0, np.ones_like(p)]])

lennard_jones_oscillator = SingleIntegrand(
    title = "Lennard-Jones Oscillator",
    x0 = np.ones(2), # (1,0) => H = 0
    hamiltonian = lennard_jones_hamiltonian,
    f = lennard_jones_f,
    df = lennard_jones_df,
    sigma=1.,
    nu = None,
)

# Plane pendulum (see e.g. LR04):
def plane_pendulum_f(x, g=9.81, L=None, d=1):
    if L is None: L = g
    return np.einsum("ij, j", J(d), [g/L * np.sin(x[0]), x[1]])

def plane_pendulum_df(x, g=9.81, L=None, d=1):
    if L is None: L = g
    return np.einsum("ij, jk", J(d), [[g/L * np.cos(x[0]), 0], [0, np.ones_like(x[1])]])

plane_pendulum = SingleIntegrand(
    title="Plane Pendulum",
    x0 = np.ones(2),
    f = plane_pendulum_f,
    df = plane_pendulum_df,
    hamiltonian = lambda x: .5 * x[...,1] ** 2 - np.cos(x[...,0]),
    nu = None,
)

# perturbed pendulum (FHP04)
def perturbed_pendulum_hamiltonian(x):
    return .5 * x[..., 1] ** 2 - np.cos(x[..., 0]) + .2 * np.sin(2*x[..., 0])

def perturbed_pendulum_f(x, d=1):
    return np.einsum("ij, j", J(d), [np.sin(x[0]) + .4 * np.cos(2 * x[0]), x[1]])

def perturbed_pendulum_df(x, d=1):
    return np.einsum("ij, j", J(d), [[np.cos(x[0]) - .8 * np.sin(2*x[0]), np.zeros_like(x[0])], [np.zeros_like(x[1]), np.ones_like(x[1])]])

perturbed_pendulum = SingleIntegrand(
    title = "Perturbed Pendulum",
    x0 = np.array([0, 2.5]),
    f = perturbed_pendulum_f,
    df = perturbed_pendulum_df,
    hamiltonian = perturbed_pendulum_hamiltonian,
    nu = None,
)



## Hooke's law with 2 degrees of freedom (Faou, Hairer, 2004 section 4.4)

## Fermi-Pasta-Ulam problem (BIT and HLW, i think)
# def fermi_pasta_ulam():
#     def H(x):
#         ...


if __name__ == "__main__":
    pass
