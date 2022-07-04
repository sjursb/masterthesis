from pytest import approx
import numpy as np
import sys
import os
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
  
# now we can import the module in the parent
# directory.
from support_functions import *

def test_calc_observed_order():
    base = 3.0
    rng = np.random.Generator(np.random.PCG64(seed=100))
    
    # standard use case, which is with increasing stepsize in calculation
    orders = 5 * np.arange(5,-1,-1)
    const = rng.integers(1,100,1)
    errors = const / base ** orders
    assert calc_convergence_order(errors, base=base, increasing=True) == approx(5)

    # Check that expression is correct up to number of decimals with decreasing error size
    base = np.e
    orders = (np.arange(7) + rng.standard_normal(size=7))
    errors = 1 / base ** orders
    decimals = 3
    assert calc_convergence_order(errors, base=base, decimals=decimals, increasing=False) == np.round(np.mean(orders[1:] - orders[:-1]), decimals=decimals)

    # Check that values beneath threshold don't affect order
    tol = 1e-6
    base = 4.2
    orders = np.array([i for i in range(10)] + list(13 + rng.standard_normal(size=8)))
    errors = 1 / base ** orders
    assert calc_convergence_order(errors, base=base, order_tol=tol, increasing=False) == approx(1)

    # Check that for no errors above tolerance the method returns np.nan
    errors = np.arange(10)
    tol = 10
    assert np.isnan(calc_convergence_order(errors, order_tol=10))


