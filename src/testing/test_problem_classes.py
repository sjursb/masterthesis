from pytest import approx
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
from problem_classes import *

# NOTE: pytest cannot have nested test functions or unit classes

def test_problems():
    pass


def test_check_power_compatibility():
    temp = BaseODE()

    assert temp.power_ref == temp.param.ref_power.default and temp.power_max == temp.param.max_power.default
    assert temp.power_ref > temp.param.max_power.default
    
    try: temp.power_ref = 1 # Try to go out of bounds
    except ValueError: temp.power_ref = temp.param.ref_power.bounds[0] # Adjust for being out of bounds
    assert temp.power_ref == temp.power_max == temp.param.ref_power.bounds[0] # Check that max_power also was changed
    assert temp.comparisons < temp.param.comparisons.default # Check that number of comparisons has changed

def test_get_dts():

    p = BaseODE(base=6, ref_power=8, max_power=5, comparisons=4)
    dt_ref, dts = p.get_dts()
    assert len(dts) == p.comparisons
    assert min(dts) == p.base ** - p.power_max
    assert dts[0] / dt_ref == p.base ** (p.power_ref - p.power_max)
    for i in range(p.comparisons-1):
        assert dts[i+1] / dts[i] == approx(p.base)

def test_get_timesteps():
    p = BaseODE(base=3, ref_power=13, max_power=8, comparisons=7, t0=0.5, T=123)

    dt_ref, dts = p.get_dts()
    timesteps_ref, timesteps_array = p.get_timesteps()
    dT = p.T - p.t0

    assert np.allclose(timesteps_array * dts, dT)
    assert timesteps_ref == approx(dT * p.base ** p.power_ref)


if __name__ == "__main__":
    pass