import os

from visualization import make_color_lists
from local_variables import paths, figure_path, table_path, data_path, xla_flag, batches, batch_simulations # differs from computer to computer

# To use cpu cores in calculations:
os.environ['XLA_FLAGS'] = xla_flag

import jax
from problems import *
from rk_builder import RK
from time import time

def deterministic_collocation_convergence(p:SingleIntegrand = harmonic_oscillator, slopes:"list[list[int]]"=[[], []]):
    """Generate convergence plot for collocation methods for rigid body and harmonic oscillator."""
    # oscilator: slopes=[]
    # rigid_bod: slopes=[2,4,6,8]
    # p.batches = len(jax.devices())
    # print(f"Number of devices accessible: {p.batches}")
    p.s_range = [1, 2, 3, 4]
    p.quadratures=[0,3]
    p.silent_max=0

    p.sigma=0.
    p.convergence_plots(path=figure_path, verbose=True, stochastic=False, slopes=slopes)


def plot_convergence(p:SingleIntegrand=henon_heiles, quadrature=0, s_range=None, max_slope=8, slope_factor=1e-1, silent_max=None, tol=1., store=True, verbose=True, stochastic=False,
    batches=None, batch_simulations=None, order_tol=1e-14):
    """Generate deterministic or stochastic convergence plot for problem p (default Hénon-Heiles) with given quadrature (default 0)."""
    if not stochastic: p.sigma= 0.
    else: # set some speedy default values
        p.batches = 4
        p.batch_simulations = 250
    if s_range is not None: p.s_range=s_range
    if silent_max is not None: p.silent_max=silent_max
    p.quadratures = [quadrature]
    if batches is not None: p.batches = batches
    if batch_simulations is not None: p.batch_simulations = batch_simulations
    if verbose: t0 = time()
    p.double_convergence_plots(figure_path=figure_path, table_path=table_path, data_path=data_path, alpha=0.01, tol=tol, verbose=verbose, max_slope=max_slope, slope_factor=slope_factor, stochastic=stochastic, mask=True, store=True, order_tol=order_tol) 
    if verbose: print(f"Finished in {np.round(time()-t0, 2)} seconds.")

def make_time_plot(p:SingleIntegrand=henon_heiles, hbvms:"list[RK]"=None, powers=None, power:int=None, T=.5, subplots=None, verbose=True, stochastic=False, legend_idx=0):
    
    if subplots is None: subplots = ["error", "Hamiltonian"]
    if hbvms is None:
        ss = [2,2,3,3]
        ks = [3,4,4,5]
        hbvms = [RK(k=k, s=s, quadrature=q) for s, k, q in zip(ss, ks, [0,3,3,0])] + [RK(s=int(np.max(ss)), quadrature=q) for q in [0,3]]
    if powers is None: 
        if power is None: power = 6
        powers = np.array([power]*len(hbvms))
    dts = 2. ** - powers

    #Adjust ref. solution:
    p.power_max = int(np.max(powers))
    p.power_ref = p.power_max +  1

    p.time_plots(hbvms, dts = dts, T=T, stochastic=stochastic, subplot_list=subplots, legend_idx=legend_idx, order=1, verbose=verbose)

def make_contour_plot(h6:bool=True, hbvms:"list[RK]"=None, powers:np.ndarray=None, power=3, T=.5, verbose=True, stochastic=False, store=False):

    if h6: 
        p = h6
        H = lambda q, p: p**3/3 - p/2 + q**6/30 + q**4/4 - q**3/3 + 1/6 # H order 6
        xlim = (-.8, 1.4)
        ylim = (0., 1.2)
        colormap = 'Purples'
        # if stochastic: colormap = 'Reds'

    else:  
        p = kubo_oscillator
        H = lambda q, p: .5 * (q**2 + p**2) # Kubo Oscillator
        xlim = ylim = (-1.1, 1.1)
        colormap = 'bone'
    
    # Double well should be PuRd

    if hbvms is None: 
        hbvms = [RK(k=6, s=2, quadrature=3)]
    if powers is None: 
        powers = np.array([power]*len(hbvms))
        # if stochastic: powers += 1
    dts = p.base ** - powers

    #Adjust ref. solution:
    p.power_max = int(np.max(powers))
    p.power_ref = p.power_max +  3

    p.H_contour_plot(H, hbvms, stochastic=stochastic, dts = dts, T=T, xlim=xlim, ylim=ylim, time_plots=False, colormap=colormap, show=True, store=store, show_legend=True)

def kepler_orbit(hbvms:"list[RK]", power:int, periods=10, verbose=True, stochastic=False, q=0):
    p = kepler_2d

    
    if stochastic: 
        power += 2
        sigma = p.sigma
    else: sigma = 0

    # Prepare sizes
    dt = 1 / p.base ** power
    p.T = periods * 2*np.pi
    p.power_max = p.power_ref = power
    p.generate_dW((p.base, power), sigma=sigma)

    # Prepare plotting
    import matplotlib.pyplot as plt
    colors = make_color_lists(hbvms)
    fig, ax = plt.subplots(dpi=256)
    ax.set(title=f"Kepler orbit, Periods=$10^{int(np.log10(periods))}$, $\Delta t = {p.base}^{{-{int(power)}}}$")
    ax.patch.set(facecolor="xkcd:dark blue", alpha=0.8)


    for hbvm, color in zip(hbvms, colors):
        x = p.sde_solver(hbvm, dt, t0=0, T=p.T, sigma=sigma).squeeze()
        plt.plot(x[..., 0], x[..., 1], label=hbvm.method_name, color=color)
    plt.plot((0),(0), marker="*", color="#00ffffff", linestyle="none", markersize=10., label="Mass center")
    ax.set(aspect="equal", xlabel="$q_1$", ylabel="$q_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Kepler orbit.png")
    plt.show()
    
def customized_convergence_plots():
    """Load large simulation data and generate convergence plots with custom featured, masking data with high uncertainty."""
    simulation_filenames = ["H order 6_sigma_1.0_v00.npz", "H order 6_sigma_1.0_v01.npz", "Kepler Problem_sigma_1.0_v00.npz", "Kepler Problem_sigma_1.0_v01.npz"]
    problems = [h6, h6, kepler_2d, kepler_2d]
    order_tols = [1e-16, 1e-16, 1e-15, 1e-15]
    slope_factors = [1e-1, 1e-1, 1, 1]
    

    for problem, filename, order_tol, slope_factor in zip(problems, simulation_filenames, order_tols, slope_factors):
        problem.double_convergence_plots(
            simulation_filename=filename, figure_path=figure_path, table_path=table_path, data_path=data_path, order_tol=order_tol,
            tol=1, alpha=0.01, mask=True, store=True, slope_factor=slope_factor, max_slope=6, verbose=True)


if __name__ == "__main__":
    pass
    # Collocation performance on rigid body and harmonic_oscillator
    # for p, slopes in zip([rigid_body, harmonic_oscillator], [None, [[], []]]):
    #     deterministic_collocation_convergence(p=p, slopes=slopes)

    # Relative convergence
    # henon_heiles.relative_convergence_plot(stochastic=False)
    
    # # Deterministic convergence plots
    # for p, slope_factor in zip([henon_heiles, h6, kepler_2d], [1e-1, 1e-1, 1]):
    #     for q in [0, 3]:
    #         plot_convergence(p=p, quadrature=q, stochastic=False, slope_factor=slope_factor, max_slope=10)
    
    # # Stochastic convergence plots
    # for p, slope_factor, order_tol in zip([henon_heiles, h6, kepler_2d], [1e-1, 1e-1, 1], [1e-15, 1e-16, 1e-15]):
    #     for q in [0,3]:
    #         plot_convergence(p=p, quadrature=q, stochastic=True, batches=batches, batch_simulations=batch_simulations, slope_factor=slope_factor, order_tol=order_tol, verbose=True)
    
    # Customized stochastic plots
    # customized_convergence_plots()

    # Oscillator time plots
    # for p, stochastic, power in zip([harmonic_oscillator, kubo_oscillator], [False, True], [3,3]):
    #     make_time_plot(p=p, T=10**3, power=power, stochastic=stochastic, legend_idx=1)

    # Contour plots
    #for hbvms, stochastic in zip([[RK(k=6, s=2, quadrature=3)], [RK(k=6, s=2, quadrature=3)], [RK(k=2, s=2, quadrature=3)]], [False, True, True]):
    # make_contour_plot(h6=True, T=10**4, hbvms=[RK(k=2,s=2,quadrature=3)], stochastic=True, power=4, store=False)


    # Kepler problem
    #hbvms = [RK(k=4, s=s, quadrature=0) for s in [2,3,4]]
    #make_time_plot(p=kepler_2d, hbvms=hbvms, T=10**3, power=8, subplots=["error", "hamiltonian", "invariant"], stochastic=True)
    # kepler_orbit([RK(s=3, k=4, quadrature=0)], power=4, periods=10**4, stochastic=False)

