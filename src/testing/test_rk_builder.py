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
from rk_builder import *

# Testing functions
def test(): # Possibly redundant
    pass

def test_P():
    assert shifted_legendre(0) == 1
    assert shifted_legendre(1, y) == 2 * y - 1
    assert expand(shifted_legendre(4)) == 70 * x ** 4 - 140 * x ** 3 + 90 * x ** 2 - 20 * x + 1

def test_get_roots():
    a = np.array([rt.n() for rt in get_roots(8)]).astype(np.float64)
    b = np.array([0,0.5012100229e-1,0.1614068602e0,0.3184412681e0,0.5000000000e0,0.6815587319e0,0.8385931398e0,0.9498789977e0,0.1e1])
    assert np.allclose(a, b)

def test_get_c():
    # Assert if first 3 gauss(type 0) abscissae are returned
    assert np.allclose([0.5], get_c(1, quadrature=0, numeric=True)) # Assert if 1 stage is midpoint 
    assert np.allclose([.5 - np.sqrt(3)/6, .5 + np.sqrt(3)/6], get_c(2, quadrature=0, numeric=True)) # Assert gauss4 abscissae
    assert np.allclose([.5 - np.sqrt(15)/10, .5, .5 + np.sqrt(15)/10], get_c(3, quadrature=0, numeric=True))

    # Assert if first 3 lobatto(type III) abscissae are returned
    assert np.allclose([0.,1.], get_c(1, quadrature=3, numeric=True))
    assert np.allclose([0., 0.5, 1.], get_c(2, quadrature=3, numeric=True))
    assert np.allclose([0., (5 - np.sqrt(5))/10, (5 + np.sqrt(5))/10, 1.], get_c(3, quadrature=3, numeric=True))

def test_lobatto_b():
    rng = np.random.default_rng(seed=100) # equivalent to np.random.Generator(np.random.PCG64(seed=100))
    k = rng.integers(20)
    b = lobatto_b(k, numeric=True)
    assert b.sum() == approx(1) # Condition of RK method

def test_gauss_b():
    for k in range(1,5):
        c = get_c(k, quadrature=0)
        assert np.allclose(collocation_b(k, c, quadrature=0, numeric=True), gauss_b(k, c,numeric=True))

def test_get_b():
    # Assert if first 3 gauss(type 0) quadrature weights are returned
    assert np.allclose([1.], collocation_b(k=1, numeric=True, quadrature=0))
    assert np.allclose([.5, .5], collocation_b(k=2, numeric=True, quadrature=0))
    assert np.allclose([5/18, 4/9, 5/18], collocation_b(k=3, numeric=True, quadrature=0))

    # TODO: Make assertions for Radau type I and II as well

    # Assert if first 3 lobatto(type III) quadrature weights are returned
    assert np.allclose([.5, .5], collocation_b(k=1, quadrature=3, numeric=True))
    assert np.allclose([1/6, 2/3, 1/6], collocation_b(k=2, quadrature=3, numeric=True))
    assert np.allclose([1/12, 5/12, 5/12, 1/12], collocation_b(k=3, quadrature=3, numeric=True))

    # Assert that lobatto_b and get_b returns same weights
    assert np.allclose(collocation_b(k=5, numeric=True, quadrature=3), lobatto_b(k=5, numeric=True))

def test_collocation_A():
    # Test Gauss method table
    assert np.allclose(collocation_A(s=1, numeric=True, quadrature=0), [0.5])
    assert np.allclose(
        collocation_A(s=2, numeric=True, quadrature=0), 
        [[1/4, 1/4 - np.sqrt(3)/6], [1/4 + np.sqrt(3)/6, 1/4]]
    )
    assert np.allclose(
        collocation_A(s=3, numeric=True, quadrature=0),
        [
            [5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
            [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
            [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
        ]
    )

    # Check greater degree against c
    s = 5
    gauss5_A = collocation_A(s, numeric=True, quadrature=0)
    assert np.allclose(get_c(k=5, numeric=True, quadrature=0), gauss5_A.sum(axis=1))
    assert np.linalg.matrix_rank(gauss5_A) == s # rank = s => order = 2s
    
    # Check for Lobatto method:
    assert np.allclose(collocation_A(s, numeric=True, quadrature=3), hbvm_A_lobatto(s,s, numeric=True))


def test_lobatto_A():
    k, s = 6,4
    A, b, c = get_Abc(k, s, numeric=True, quadrature=3)
    
    assert np.linalg.matrix_rank(A) == s # Checks that it achieves correct order
    assert np.allclose(A[-1], b) # checks if stiffly accurate

def test_hbvm_A_gauss():
    k, s = 3,3
    assert np.allclose(collocation_A(s, quadrature=0, numeric=True), hbvm_A_gauss(s,s, numeric=True))
    k = 5
    assert np.linalg.matrix_rank(hbvm_A_gauss(k, s, numeric=True)) == s

def test_hbvm_A():
    s = 2
    for k in range(2,6):
        assert np.allclose(hbvm_A(k, s, numeric=True, quadrature=3), hbvm_A_lobatto(k, s, numeric=True))
        assert np.allclose(hbvm_A(k, s, numeric=True, quadrature=0), hbvm_A_gauss(k, s, numeric=True))

def test_rk_hbvm_lobatto(): 
    # Testing method class
    # Check if Lobatto 4 has rank 2
    s = 2
    assert np.linalg.matrix_rank(RK(s=s, numeric=True).A) == s 
    
    # Compare results with some Lobatto IIIA methods
    trapezoid = RK(A=np.array([[0,0], [.5, .5]]), b=np.array([.5, .5]), c=np.array([0.,1.]))
    assert trapezoid.is_same_method(RK(s=1))
    lob4 = RK(A=np.array([[0,0,0], [5/24, 1/3, -1/24], [1/6, 2/3, 1/6]]), b = np.array([1/6, 2/3, 1/6]), c=np.array([1/6, 2/3, 1/6]))
    assert lob4.is_same_method(RK(s=2, quadrature=3))

def test_store_as_npz():
    filename = "test_file"
    assert os.path.exists(filename) is False
    
    rk = RK(k=4, s=3, quadrature=0, numeric=True)
    rk.store_as_npz(filename)
    filename += ".npz"
    assert os.path.exists(filename)

    data = np.load(filename)
    A, b, c = data['A'], data['b'], data['c']
    data.close()
    assert np.array_equal(rk.A, A)
    assert np.array_equal(rk.b, b)
    assert np.array_equal(rk.c, c)
    
    os.remove(filename)
    assert os.path.exists(filename) is False

def test_load_from_npz():
    filename = "test_file"
    rk1 = RK(k=5, s=2, quadrature=3, numeric=True)
    rk1.store_as_npz(filename)
    
    rk2 = RK(k=1, s=1, quadrature=0)
    filename += ".npz"
    assert rk2.load_from_npz(filename)

    for array1, array2 in zip([rk1.A, rk1.b, rk1.c], [rk2.A, rk2.b, rk2.c]):
        assert np.array_equal(array1, array2)
    os.remove(filename)

def test_symplectic(k=4, s=2): # not true test function
    rk = RK(k=k, s=2)
    B = np.diag(rk.b)
    try: 
        assert np.allclose(rk.A.T@B + B@rk.A - rk.b @ rk.b.T, 0)
        print("The method is symplectic.")
    except AssertionError:
        print("HBVM({},{}) is not symplectic".format(k, s))

if __name__=="__main__":
    test_lobatto_A()