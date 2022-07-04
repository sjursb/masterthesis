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
from visualization import *

def test_get_dW_list():
    t0=0; T=5; base=3; max_power=6; batch_simulations=10**3; batches=20; comparisons=2; diffusion_terms=4; seed=101

    dW_list = get_dW_list(t0=t0, T=T, base=base, power_max=max_power, comparisons=comparisons, diffusion_terms=diffusion_terms, seed=seed, batches=batches, batch_simulations=batch_simulations)
    assert len(dW_list) == comparisons

    dts = 1 / base ** np.arange(max_power, max_power-comparisons, -1)
    for i in range(comparisons):
        timesteps = int((T-t0)/dts[i])
        
        assert dW_list[i].shape == (timesteps, batches, batch_simulations, diffusion_terms)
        assert np.mean(dW_list[i]) == approx(0, abs=5e-5)
        assert dW_list[i].std() == approx(np.sqrt(dts[i]), rel=1e-2)
