# plotting functions used in master thesis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np

from rk_builder import RK
from support_functions import assess_scale

def k_plot(self, e, labels, de=None, base_max_ref=None, slope_factor=None, show=False,): #TODO: Rewrite using ax_line_plot
    if base_max_ref is None: base_max_ref = (self.base, self.power_max, self.power_ref)
    base, power_max, power_ref = base_max_ref
    silent_stages = range(self.silent_max + 1)
    dt = 1 / base ** power_max
    fig = plt.figure(self.title + " k-plot base {} power {}".format(base, power_max))
    plt.title(self.title + R" k convergence plot for $\Delta t = {}^{{-{}}}$".format(base, power_max))
    plt.yscale("log")#, base=base) # log2-scale seems weird, for some reason
    for i in range(len(e)):
        print(e[i])
        if de is None: plt.plot(silent_stages, e[i], label=labels[i])
        else: plt.errorbar(silent_stages, y=e[i], yerr=de[i], label=labels[i], capsize=3)
    if slope_factor is None: slope_factor = np.mean([error[0] for error in e])
    for i in range(2,7,2): plt.plot(silent_stages, slope_factor * dt ** (i*np.array(silent_stages)), label=R"$\mathcal{{O}}(\Delta t^{{{}k}})$".format(i), linestyle="dashed")
    plt.legend()
    plt.xticks(silent_stages)
    plt.xlabel("Silent stages")
    plt.ylabel("Error")
    plt.xlim(0, self.silent_max)
    plt.ylim(.9*e.min(), 1.1*e.max())
    if show: plt.show()
    else: return fig

def convergence_plot(title, hbvms:"list[RK]", dts, e, de=None, slopes=None, slope_factor=1e-1, ylabel="Error", legend_outide=True):

    # hbvms errors and error_deviations are assumed of same length
    # dts, errors[i] and error_deviations[i] are assumed of same length
    if slopes is None: slopes = np.arange(hbvms[0].s, hbvms[-1].k)
    if len(slopes): hbvm_colors, slope_colors = make_color_lists(hbvms, slopes)
    else: 
        hbvm_colors, slope_colors = make_color_lists(hbvms), []

    # TODO: rewrite to match double_convergence_plot (or merge using strong=[bool, bool] or something)
    fig = plt.figure(title)
    plt.title(title)
    labels = []
    for i in range(len(hbvms)):
        label = hbvms[i].method_name
        labels.append(label)
        
    for i in range(len(hbvms)):
        if de[i] is None: plt.loglog(dts, e[i], label=labels[i], color=hbvm_colors[i], marker='x')
        else: plt.errorbar(dts, e[i], de[i], label=labels[i], capsize=3, color=hbvm_colors[i])
    
    for s, color in zip(slopes, slope_colors):
        plt.loglog(dts, (slope_factor * dts)**s, label=r"Slope order {}".format(s), linestyle="dashed", color=color)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Step length")
    plt.ylabel(ylabel)
    plt.xlim(dts.min(), dts.max())
    plt.ylim(bottom=1e-17)

    return fig

def get_slopes(hbvms:"list[RK]", nu, stochastic:bool=False):
    """Get slopes matching convergence order of SingleIntegrand HBVM solvers."""
    s_min = np.min([hbvm.s for hbvm in hbvms])
    s_max = np.max([hbvm.s for hbvm in hbvms])
    k_max = np.max([hbvm.k for hbvm in hbvms])
    if nu is None: k_max += 1
    slopes = np.array([np.arange(s_min, k_max), np.arange(s_min, s_max+1)], dtype=object)
    if stochastic: return slopes
    else: return 2 * slopes


def ax_line_plot(ax:plt.Axes, plot_dict:dict, plot_items:list, alpha=0.2):
    """Decorate a subplot ax with specified plot_dict and plot_items with different x-lists."""
    ax.set(**plot_dict)
    for x, y, yerr, fill_between, param_dict in plot_items:
        if yerr is None: # No confidence interval
            ax.plot(x,y, **param_dict)
        elif fill_between: # Show confidence as shaded area around plotted line
            ax.plot(x, y, **param_dict)
            param_dict["label"] = None # To avoid duplicate labels
            ax.fill_between(x, y-yerr, y+yerr, alpha=alpha, **param_dict) # aplha sets visibility of fill
        else: # Plot line with error bars
            ax.errorbar(x, y, yerr=yerr, capsize=3, **param_dict)

    return None

def time_plot(
        hbvms:"list[RK]", t0:float, T:float, title:str="", ys:"list[np.ndarray]"=[], 
        yerrs:"list[np.ndarray]"=None, y_exact:np.ndarray=None, fill_between:"list[bool]"=None, base=2.,
        ax:"plt.Axes"=None, ylabel="", store=True, scale=["linear", "log"], ylim:"tuple[float, float]"=None, show_legend=True
    ):
    """Plot a list of approximations ys generated by respective methods in hbvms and and exact solution y_exact against time t."""
    # TODO: expand doc to explain functionality and test
    # Preliminary sizes
    K = len(hbvms)
    
    labels = [hbvms[i].method_name for i in range(K)]
    colors = make_color_lists(hbvms)
    linestyles = ["-" for _ in range(K)] # Maybe supply option?
    ts =  [np.linspace(t0, T, len(y)) for y in ys]

    if y_exact is not None: # add features of exact solution to respective lists
        ys.append(y_exact); labels.append("Exact solution"); colors.append("k"); linestyles.append("-"); ts.append(np.linspace(t0, T, len(y_exact)))

    dts = [t[1]-t[0] for t in ts]
    if not np.allclose(np.full_like(dts, dts[0]), dts): # If stepsizes are different
        for i in range(len(labels)): labels[i] += R", $\Delta t = {}^{{-{}}}$".format(int(base), int(np.log(dts[i])/np.log(base)))

    if yerrs is None: yerrs = np.full(len(ys), None)
    else: yerrs.append(None) # Exact has no error, of course

    # Plot features
    if fill_between is None: fill_between = np.full(len(ys), None)
    scale = assess_scale(scale)
    
    if ylim is None: 
        bottom, top = np.min([np.min(y) for y in ys]), np.max([np.max(y) for y in ys])
        if scale[1] == 'log': bottom = max(bottom, 1e-17)
        ylim = (bottom, top)
    if not len(ylabel): ylabel = "x(t)"
    
    plot_dict = dict(title=title, xlim=(t0, T), ylim=ylim, xlabel="t", ylabel=ylabel, xscale=scale[0], yscale=scale[1])
    
    if not show_legend: labels = [None for _ in range(len(labels))]
    plot_items = [[ts[i], ys[i], yerrs[i], fill_between[i], dict(color=colors[i], label=labels[i], linestyle=linestyles[i])] for i in range(len(ys))]
    # The plotting
    if ax is None: 
        if store: layout = 'constrained'
        else: layout = None
        fig, ax = plt.subplots(layout=layout, figsize=(8,4))
    else: fig = None
    ax_line_plot(ax, plot_dict, plot_items, plot_items)
    if show_legend: ax.legend()#loc="upper left", bbox_to_anchor=(1., 1.))
    return fig, ax


def ax_convergence_plot(ax:plt.Axes, plot_dict, plot_items, x):
    """Decorate a subplot ax with specified plot_dict and plot plot_items using the same x-list"""

    ax.set(**plot_dict) #title, xlabel, ylabel, xscale, yscale, ylim, etc. -> see plt.set() or matplotlib.Axes.set()
    for y, yerr, param_dict in plot_items:
        if yerr is None: ax.plot(x, y, **param_dict) # color, label, linestyle
        else: ax.errorbar(x, y, yerr=yerr, capsize=3, **param_dict)
    ax.set_xlim(np.min(x), np.max(x))
    
def double_convergence_plot(
        title, hbvms:"list[RK]", dts, e, de=None, slopes=None, slope_factor=1e-1, max_slope=8,
        global_error=True, store=True, stochastic=False, double_legend=False
    ):
    """
    double_convergence_plot Makes convergence plot with Hamiltonian and ODE as subplots with common legend

    _extended_summary_

    Parameters
    ----------
    title : str
        Title displayed as suptitle
    hbvms : list[RK]
        Methods tested for convergence corresponding with elements in e, de
    dts : np.ndarray
        step-sizes corresponding to errors in e, de
    e : np.ndarray
        errors of the respective hbvms
    de : _type_, optional
        error deviation (if SDE), by default None
    slopes : list, optional
        double list of slope orders to display in left and right plot, by default None
    slope_factor : float, optional
        Number to adjust initial value of slopes, by default 1e-1
    global_error : bool, optional
        Whether error is global error or not (adjusts slopes), by default True
    store : bool, optional
        Whether to store image or not (adjusts some values in plot configuration), by default True
    stochastic : bool, optional
        Whether or not stochastic plot (gives errorbars), by default False

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Double convergence plot figure
    """
    if de is None: 
        de = np.full(e.shape[:-1], None)
    if slopes is None: slopes=[[],[]]
    hbvm_colors, slope_colors = make_color_lists(hbvms, slopes[0]) # slopes[0] will always be the largest one

    subtitle = [["Hamiltonian", "Solution"], ["Weak/Hamiltonian", "Strong/Mean Square"]][stochastic]
    if store: layout = 'constrained'
    else: layout = None # Constrained is a good starting point, but breaks down once changes are made interactively
    fig, axs = plt.subplots(1,2, layout=layout, figsize=(8,4))
    ylabels = [
        [R"$|H(x_{{approx}})-H(x_{{exact}})|$", R"$||x_{{exact}} - x_{{approx}} ||_2$"],
        [R"$\left|E\left[H(x_{{approx}})\right] - E\left[H(x_{{exact}})\right]\right|$", R"$\sqrt{{E\left[||x_{{exact}} - x_{{approx}} ||_2^2\right]}}$"]
    ][stochastic]
    marker = ["x", None][stochastic]
    labels = [[hbvms[i].method_name, None] for i in range(len(hbvms))]
    fig.suptitle(title)
    for strong in range(2):
        # Set up subplot plot_dict
        floor = [[1e-17, 1e-16], [1e-16, 1e-15]][strong][global_error]
        ylim = (floor, np.max(e[strong]))
        plot_dict = dict(title=subtitle[strong], xlabel=r"$\Delta t$", ylabel=ylabels[strong], xscale='log', yscale='log', ylim=ylim)
        
        # Set up subplot plot_items (ie hbvm values[, errors], colors, labels)
        plot_items = [[e[strong, i], de[strong, i], dict(color=hbvm_colors[i], label=labels[i][strong], marker=marker)] for i in range(len(hbvms))]
        # set up slopes
        temp_slopes = slopes[strong]
        for j in range(len(temp_slopes)):
            s = temp_slopes[j]
            i = int(not global_error)
            if double_legend: # If legend is used for both local and global error
                slope_label = [f"Slope order {s} + i", None][strong] # only labels on first subplot - the rest have the same colors
            else:
                slope_label = [f"Slope order {s}", None][strong]
            slope_features = [(slope_factor * dts)**(s+i), None, dict(color=slope_colors[j], linestyle="dashed", label=slope_label)]
            if s <= max_slope: plot_items.append(slope_features)
            else: pass

        # Do the plotting
        ax_convergence_plot(axs[strong], plot_dict, plot_items, x=dts)
    # Add legends
    if store: bbox_to_anchor = (-0.2, -0.2 -.1 * (len(plot_items)//4))
    else: bbox_to_anchor = None
    if global_error: # Only show figlegend in global error plot
        fig.legend(ncol=4, loc="lower center", bbox_to_anchor=bbox_to_anchor, bbox_transform=axs[1].transAxes)

    return fig

def relative_convergence_plot(title:str, hbvms:"list[RK]", dts:np.ndarray, relative_errors:np.ndarray, s_range:"list[int]", marker="x", store=False):

    labels = [hbvm.method_name for hbvm in hbvms]
    hbvm_colors = make_color_lists(hbvms=hbvms)

    n_plots = len(s_range) # One subplot for each order of convergence

    fig, axs = plt.subplots(n_plots, 1, sharex=True)
    fig.suptitle(title)

    for i in range(len(hbvms)):
        hbvm = hbvms[i]
        s = hbvm.s
        if s != hbvm.k and hbvm.quadrature == 3: linestyle = "dashed"
        else: linestyle = "solid"
        
        axs[s-1].plot(dts, relative_errors[i], color=hbvm_colors[i], label=hbvm.method_name, linestyle=linestyle, marker=marker)
    
    
    for s in s_range:
        ax:plt.Axes =axs[s-1]
        ax.set(title=f"s={s}, Gauss-{2*s} reference", ylabel=R"Relative error".format(s, 2*s))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.set(xlabel=R"$\Delta t$", xscale="log")
    plt.figlegend(loc='center right', bbox_to_anchor=(1, 0.5))

    return fig


    


# For reference:
colorschemes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 
'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 
'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 
'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 
'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r',
'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 
'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 
'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

def make_color_lists(hbvms:"list[RK]"=[], slope_list:"list[int]"=[], cmaps=dict(gauss='Blues', lobatto='Reds', slope='binary', g_hbvm='spring', l_hbvm='cool')):
    """Make lists of colors from different cmaps based on quadrature corresponding to hbvms and slope_list"""
    gauss_cmap = plt.get_cmap(cmaps['gauss'])
    lobatto_cmap = plt.get_cmap(cmaps['lobatto'])
    slope_cmap = plt.get_cmap(cmaps['slope'])
    g_hbvm_cmap = plt.get_cmap(cmaps['g_hbvm'])
    l_hbvm_cmap = plt.get_cmap(cmaps['l_hbvm'])
    n_gauss, n_lobatto, n_g_hbvm, n_l_hbvm = 0, 0, 0, 0
    for hbvm in hbvms:
        if hbvm.quadrature == 0: 
            if hbvm.k == hbvm.s: n_gauss += 1
            else: n_g_hbvm += 1
        elif hbvm.quadrature == 3: 
            if hbvm.k == hbvm.s: n_lobatto += 1
            else: n_l_hbvm += 1
    gauss_colors = [gauss_cmap(x) for x in np.linspace(0.9, .3, n_gauss)]
    lobatto_colors = [lobatto_cmap(x) for x in np.linspace(0.9, .3, n_lobatto)]
    g_hbvm_colors = [g_hbvm_cmap(x) for x in np.linspace(0,1, n_g_hbvm)]
    l_hbvm_colors = [l_hbvm_cmap(x) for x in np.linspace(0, 1, n_l_hbvm)]
    hbvm_colors=[]
    for hbvm in hbvms:
        if hbvm.quadrature == 0: 
            if hbvm.k == hbvm.s: hbvm_colors.append(gauss_colors.pop())
            else: hbvm_colors.append(g_hbvm_colors.pop())
        elif hbvm.quadrature == 3: 
            if hbvm.k == hbvm.s: hbvm_colors.append(lobatto_colors.pop())
            else: hbvm_colors.append(l_hbvm_colors.pop())
        

    slope_colors = [slope_cmap(x) for x in np.linspace(0.2, 1., len(slope_list))]
    if not len(slope_list): return hbvm_colors
    elif not len(hbvm_colors): return slope_colors
    else: return hbvm_colors, slope_colors



def contour_plot(fig:plt.Figure, ax:plt.Axes, f:callable, plot_dict=dict(xlim=(-1,1), ylim=(-1,1)), lines:list=None,
        f_name=None, zlim=(-1,1), 
        n_levels=10, num=10**3, cmap=cm.get_cmap('Purples')
    ):
    """Make a countour plot on given axis based on input features"""
    
    ax.set(**plot_dict)
    # set up sizes
    x = np.linspace(*plot_dict["xlim"], num=num)
    y = np.linspace(*plot_dict["ylim"], num=num)

    xx, yy = np.meshgrid(x, y)
    zz = f(xx, yy) # todo: add masking? - make sure it works with hamiltonian taking only one element!

    if zlim is None: zlim = (np.min(zz), np.max(zz))
    levels = np.linspace(*zlim, num=n_levels) # alt. use logspace

    zlim = np.array(zlim) # for safety
    zz = np.ma.masked_where(zz < zlim[0], zz) # mask below
    zz = np.ma.masked_where(zz > zlim[1], zz) # mask above

    
    contset = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    contlines = ax.contour(contset, levels=levels, colors='white', linestyles='dashed', alpha=.5, zorder=2.1) # Draw after lines!
    colbar = fig.colorbar(contset, ax=ax)
    colbar.ax.set_ylabel(f_name, rotation=270)# loc='top', rotation=0)
    colbar.add_lines(contlines)

    # Plot lines
    if lines is not None:
        for line, line_dict in lines: ax.plot(line[0], line[1], **line_dict) # line_dict includes label, color, linestyle, marker, 

    return fig, ax


if __name__ == "__main__":
    f  = lambda q, p: p**3/3 - p/2 + q**6/30 + q**4/4 - q**3/3 + 1/6
    #f = lambda q, p: .5 * p**2 - 1 / q ** 2 # 1d central force
    # Kepler problem
    x0 = np.array([0,1,2,0])
    #f = lambda x1, x2: (np.linalg.norm(x0[2:], axis=0, ord=2) - 1 / np.linalg.norm([x1, x2], ord=2, axis=0)) # Wrong - see hlw or hw!
    
    f = lambda q, p: 0.25 * q ** 4 - 0.5 * q**2 + 0.5 * p**2
    H = lambda q, p: .5 * (q**2 + p**2)
    plot_dict = dict(title="Hamiltonian", xlim=(-2,2), ylim=(-2,2), xlabel="q", ylabel="p")
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    golden = np.array([np.exp(-theta)*np.cos(theta), np.exp(-theta)*np.sin(theta)])
    line_list = [
        [circle, dict(color='r', linestyle='dashed', label="unit circle")],
        [golden, dict(color='k', linestyle='dotted', label="golden ratio")]
    ]

    fig, ax = plt.subplots()
    fig, ax = contour_plot(fig, ax, H, f_name="H(x)", plot_dict=plot_dict, lines=line_list, levels=30, cmap=cm.get_cmap('Blues'), zlim=(-5,5))
    plt.show()
