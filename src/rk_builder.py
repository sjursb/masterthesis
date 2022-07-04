# Calculating coefficients of HBVM methods using symbolic python
# Code is based on "Analysis of Hamiltonian Boundary Value Methods" by Brugnano et al. (2015)
import os  # For using different paths for storage
import time
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import param
from sympy import *  # Import sympy namespace in its entirety
from sympy.abc import x, y, z
import sympy.integrals.quadrature as q # For calculating gauss and lobatto weights quickly

# Make method class
class RK(param.Parameterized):
    """Class for storing Butcher table of an arbitrary Runge-Kutta method"""
    method_name = param.String(default="", doc="Name of RK-method")
    A = param.Array(doc='A coefficient matriz')
    b = param.Array(doc='b coefficients of method')
    c = param.Array(doc='c coefficients (abscissae) of method')
    # p = param.Integer(doc='Order of method') # Possibly superfluous
    
    # For constructing method from HBVM or collocation framework
    s = param.Integer(doc='Rank of matrix A')
    k = param.Integer(doc='Number of stages in RK method')
    quadrature = param.Selector(default=0, objects=[0,3], doc='Type of method: 0=Gauss, 3=Lobatto, 4=HBVM')
    numeric = param.Boolean(default=True, doc="Info about whether the Butcher table is stored as number or symbolic objects (using sympy).")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.k < self.s:
            self.k = self.s
        if self.quadrature != 0 and self.quadrature != 3:
            self.quadrature = 3
        self.set_name(self.method_name)
        if self.A is None:
            if self.load_from_npz(): pass
            # TODO: make it possible to build from supplied b's and c's
            # TODO: Make it possible to build method from method_name
            else: self.hbvm(self.k, self.s, numeric=self.numeric, quadrature=self.quadrature)

    @param.depends("k", "s", "quadrature")
    def set_name(self, name="", override_name=False):
        
        if name:
            self.method_name = name
        elif override_name or not self.method_name:
            method_dict = {0:"Gauss", 3:"Lobatto"}
            if self.k == self.s:
                self.method_name = r"{}-{} or HBVM({},{})".format(method_dict[self.quadrature], 2*self.k, self.k, self.k)
            else:
                self.method_name = r"{}-HBVM({},{})".format(method_dict[self.quadrature], self.k, self.s)
        else:
            print("Name already set - supply name or override_name=True to change!")
        

    def hbvm_lobatto(self, k, s, numeric=True):
        """Build Hamiltonian Boundary Value method with k stages and rank s using Lobatto quadrature."""
        self.k, self.s, self.A, self.b, self.c = k, s, *get_Abc(k, s, numeric, quadrature=3)

    def hbvm_gauss(self, k, s, numeric=True):
        """Build Hamiltonian Boundary Value method with k stages and rank s using Gauss quadrature."""
        self.k, self.s, self.A, self.b, self.c = k, s, *get_Abc(k, s, numeric, quadrature=0)

    def hbvm(self, k, s, numeric=True, quadrature=None):
        if quadrature is None:
            quadrature = self.quadrature
        if quadrature == 0:
            self.hbvm_gauss(k, s, numeric=numeric)
        elif quadrature == 3:
            self.hbvm_lobatto(k, s, numeric=numeric)
        else:
            self.hbvm_lobatto(k, s, numeric=numeric)

    def lobattoIIIa(self, s, numeric=True):
        """Build Lobatto IIIa method with s stages (uses hbvm method internally)."""
        self.hbvm(s, s, numeric=numeric)
    
    def collocation(self, s, quadrature=None, numeric=None):
        """Build Butcher table for collocation method of order 2s with type 0 or III quadrature."""
        if numeric is None: numeric = self.numeric
        if quadrature is None: quadrature = self.quadrature

        self.k, self.s = s, s
        self.A = collocation_A(s, numeric, quadrature=quadrature)
        self.b = collocation_b(s, numeric=numeric, quadrature=quadrature)
        self.c = get_c(s, numeric=numeric, quadrature=quadrature)

    
    def is_same_method(self, other_method):
        """
        Check if self instance and other instance of same class are same method (both are assumed numeric)
        """
        return (np.allclose(self.A, other_method.A) and np.allclose(self.b, other_method.b), np.allclose(self.c, other_method.c))

    def write_to_tex(self, filename:"str|None"=None, path=".", numeric=True, mode="w", verbose=True):
        """Write Butcher table to tex file. NB! Overwrites old file if it exists, so be careful."""
        if filename is None:
            filename = f"{self.method_name}.tex"
        elif filename[-4:] != ".tex":
            filename +=".tex"
        else: pass
        filename = f"{path}/{filename}"

        A, b, c = self.A, self.b, self.c
        latex_output = [latex(Matrix(A)), latex(Matrix(b).T), latex(Matrix(c))]
        with open(filename, mode) as texfile:
            texfile.write('\n' + self.method_name + '\n')
            for line, variable in zip(latex_output, ['A = ', 'b = ', 'c = ']):
                texfile.write(variable + line + '\n')

        if verbose: print("Butcher table written to file '{}'".format(filename))
        return None
    
    def store_as_npz(self, filename:str=None, path=".", overwrite=False, verbose=True):
        """Store Butcher table A, b, c in `filename.npz`, with option to overwrite existing files (default False)."""
        if filename is None: filename = f"{self.method_name}"
        filename = f"{path}/{filename}"
        if os.path.exists(filename+".npz") and overwrite is False: 
            if verbose: print(f"The method is already stored in file {filename}!")
        else: 
            np.savez_compressed(filename, A=self.A, b=self.b, c=self.c)
            if verbose: print("Butcher table stored in file '{}'".format(filename))
        return None
    
    def load_from_npz(self, filename:str=None, path="."):
        """Load arrays A, b, and c from `filename.npz` file."""
        if filename is None: filename = f"{path}\{self.method_name}.npz"
        if os.path.exists(filename):
            data = np.load(filename)
            self.A, self.b, self.c = data['A'], data['b'], data['c']
            data.close()
            return True
        else: 
            return False

def shifted_legendre(n, x=symbols('x'), sympy=True):
    """Calculate shifted (but not normalized!) Legendre polynomials of order n"""
    if sympy:
        return legendre(n, 2 * x - 1)
    if n == 0:
        result = (1)
    elif n == 1:
        result = (2 * x - 1)
    else:
        result = (((2 * n - 1) * (2 * x - 1) * shifted_legendre(n - 1, x) - (n - 1) * shifted_legendre(n - 2, x)) / n)
    return result


def lagrange(i, c, x=symbols('x')):
    """Calculates Lagrange polynomial based on vector c and index i"""
    if type(c) == list: c = np.array(c)
    nominator = prod(x - c[c!=c[i]])
    denominator = prod(c[i] - c[c!=c[i]])
    return expand(nominator/denominator)

def get_roots(n, quadrature:int=3)->List[CRootOf]:
    """
    Get roots of Gauss or Lobatto polynomials of degree n.

    Identical output to fsolve(f(x) = 0, x) in Maple for respective polynomials."""
    if quadrature==0: # Gauss quadrature
        f:Poly = Poly(shifted_legendre(n, x), x)
    # elif quadrature==1 or quadrature==2: # Radau quadrature
    # TODO: Add way to make Radau I & II roots as well

    elif quadrature==3: # Lobatto quadrature
        f:Poly =Poly((x**2 - x) * diff(shifted_legendre(n, x)), x)
    else:
        Exception()
    # root_dict = polys.polyroots.roots(f, x)
    return f.all_roots()


# Alternative function definitions
gauss_roots = lambda n: get_roots(n, quadrature=0)
lobatto_roots = lambda n: get_roots(n, quadrature=3)

def get_c(k:int, numeric:bool=False, quadrature:int=3):
    """
    Calculates c column vector in Butcher table from Gauss or Lobatto polynomials.
    ### Notes:
    - Lobatto has k + 1 stages, whereas Gauss has k
        - This ensures that both quadrature have order 2k
    """
    rts:List[CRootOf] = get_roots(k, quadrature=quadrature) # False returns Lobatto nodes
    if numeric:
        return np.array([rt.n() for rt in rts]).astype(np.float64)
    else:
        return np.array(rts)

def gauss_b(k, c:"np.ndarray|None"=None, numeric:bool=False)->np.ndarray:
    """
    Calculate b coefficients in Butcher table by an explicit formula (for Gauss quadrature)
    
    Note: This construction is hopefully significantly faster than the more general collocation_b.
    """
    # Using pure Legendre expressions
    # c = sorted(solve(legendre(k, x), x))
    # P  = diff(legendre(k, x), x)
    # b = [simplify(1 / ((1 - c_i**2) * P.subs(x, c_i)**2)) for c_i in c]
    # Using shifted Legendre
    if c is None: 
        c = get_c(k, quadrature=0)
    P = diff(shifted_legendre(k, x), x)
    b = [simplify(1 / (c_i * (1 - c_i) * P.subs(x,c_i)**2)) for c_i in c]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)

def lobatto_b(k, c:"np.ndarray|None"=None, numeric:bool=False)->np.ndarray:
    """
    Calculate b coefficients in Butcher table by an explicit formula (for Lobatto quadrature)
    
    Note: This construction is significantly faster than the more general collocation_b.
    """
    if c is None: c = get_c(k, numeric=numeric, quadrature=3)
    b:List[Rational] = [1 / (k*(k+1)*shifted_legendre(k, c[i])**2) for i in range(k+1)]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)

def collocation_b(k, c:"np.ndarray|None"=None, numeric:bool=False, quadrature=0)->np.ndarray:
    """
    Calculates b from B(s) condition, which is solved as a linear system.
    
    ### Notes: 
    - Only works for Gauss and Lobatto quadrature
    - Lobatto has k + 1 stages, whereas Gauss has k
    """
    
    if c is None: c = get_c(k, quadrature=quadrature)
    if quadrature==3: k += 1 # Adjust for Lobatto IIIA having more stages

    V = Matrix([[c[j]**i for j in range(k)] for i in range(k)])
    a = Matrix([[Rational(1, q)] for q in range(1, k+1)])
    b = list(linsolve((V,a)).args[0]) # linsolve returns the solution as a Tuple within a FiniteSet, which must be unpacked and made a list - yeah, it's weird...
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)

def lagrange_b(k, c:"np.ndarray|None"=None, numeric:bool=False, quadrature=0)->np.ndarray:
    # NOTE: Untested! (and redundant, probably)
    if c is None: c = get_c(k, quadrature=quadrature)
    b = [integrate(lagrange(i, c, x), (x,0,1)) for i in range(len(c))]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)

def get_b(k, c:"np.ndarray|None"=None, numeric:bool=False, quadrature=0)->np.ndarray:
    if  (quadrature == 0 or quadrature == 3) and numeric:
        b, _ = fast_quadrature(k, quadrature)
    elif quadrature == 0: return gauss_b(k, c, numeric)
    elif quadrature == 3: return lobatto_b(k, c, numeric)
    else: return collocation_b(k, c, numeric, quadrature)


def collocation_A(s, numeric:bool=False, quadrature=0):
    """Build collocation method A coefficient matrix of rank s and declare whether numeric or symbolic"""
    c = get_c(s, quadrature=quadrature, numeric=numeric) # Adding numeric makes it a lot faster for large s
    k = len(c) # s for gauss(type 0), s+1 for lobatto(type 3)
    # Construct by integral of lagrange polynomial (see Geometric Numerical Integration pp.31)
    l = lambda i, x: lagrange(i, c, x)
    A = Matrix(k, k, lambda i, j: expand(integrate(l(j, x), (x, 0, c[i]))))

    if numeric:
        A = np.array([[A[i,j].n() for j in range(k)] for i in range(k)]).astype(np.float64)
    return np.array(A)


def fast_quadrature(k, quadrature=0):
    """Returns numeric b, c vectors constructed using built-in sympy routines. (Much faster than earlier implementation!)"""
    if quadrature==0:
        c, b = q.gauss_legendre(k, 16)
    elif quadrature == 3:
        c, b = q.gauss_lobatto(k + 1, 16)
    c:np.ndarray = (np.array(c, dtype=np.float64)+1)/2
    b:np.ndarray = np.array(b, dtype=np.float64)/2
    return b, c
# TODO: construct b and c using fast_quadrature everywhere possible!

def hbvm_A(k, s, numeric:bool=False, quadrature=0, b=None, c=None):
    """Build HBVM(k,s) coefficient matrix A with Gauss quadrature and declare whether numeric."""
    if b is None or c is None:
        if numeric:
            b, c = fast_quadrature(k, quadrature=quadrature)
        else:
            c = get_c(k, numeric=numeric, quadrature=quadrature) # Should not use numeric here for precision
            b = get_b(k, c=c, numeric=numeric, quadrature=quadrature) # Should not use numeric here for precision
    else: pass
    k = len(c) # Adjusts for lobatto having k + 1 stages 
    B = diag(b.tolist(), unpack=True)
    D = diag([(2*j + 1) for j in range(s)], unpack=True) # scaling matrix - ensures normalization

    I_bar = Matrix(k, s, lambda i,j: integrate(shifted_legendre(j, x), (x, 0, c[i])))
    W = Matrix(k, s, lambda i,j: shifted_legendre(j, c[i]))

    
    A = expand(I_bar @ D @ W.T @ B) # expand(I_bar @ D @ W.T @ B) without scaling shifted_legendre might also work - less expensive computation
    if numeric:
        A2 = [[A[i,j].n() for j in range(k)] for i in range(k)]
        return np.array(A2).astype(np.float64) # Doesna work?
    else:
        return np.array(A)

hbvm_A_lobatto= lambda k, s, numeric: hbvm_A(k, s, numeric, quadrature=3)
hbvm_A_gauss= lambda k, s, numeric: hbvm_A(k, s, numeric, quadrature=0)


def get_Abc(k, s, numeric=True, quadrature=0):
    """Shorthand function for generating A, b and c from formulas for HBVM(k,s)."""
    if numeric and (quadrature == 0 or quadrature ==3):
        b, c = fast_quadrature(k, quadrature)
    else:
        if k>3 and quadrature == 0 and not numeric:
            numeric_c = True
        if k>4 and quadrature == 3 and not numeric:
            numeric_c = True
        else:
            numeric_c = numeric
        c = get_c(k, numeric_c, quadrature)
        b = get_b(k, c, numeric, quadrature)
    A = hbvm_A(k, s, numeric, quadrature, b, c)
    
    return A, b, c

def plot_time_get_b(k_max=10, numeric=True, symbolic=False):
    ks = np.arange(1,k_max+1)
    js = []
    if symbolic: js += [0,1]
    if numeric: js += [2,3]
    times = np.zeros((len(js), k_max))
    for i in range(len(js)):
        for k in ks:
            t0 = time.time()
            if js[i]%2:
                lobatto_b(k, numeric=js[i]//2)
            else:
                collocation_b(k, numeric=js[i]//2, quadrature=3)
            times[i, k-1] = time.time() - t0
    
    plt.figure()
    plt.title("Lobatto b vs general b")
    for i in range(len(js)):
        label = ""
        if js[i]%2: label += "lobatto_b"
        else:   label += "get_b"
        label += ", numeric={}".format(bool(js[i]//2))
        plt.plot(ks, times[i], label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if False:
        s = 2
        k = s # k=s leads to collocation method
        quadrature = 0
    if True: 
        # store all the necessary hbvm tables as .tex and .npz files
        quadratures = [0, 3]
        s_range = range(1, 5+1)
        
        quad_dict = {0:"Gauss", 3:"Lobatto"}
        for quadrature in quadratures:
            for s in s_range:
                silent_max = 3*s # nu * s / 2, nu=6
                for r in range(silent_max+1):
                    k = s + r
                    tex_filename = f"{quad_dict[quadrature]}-HBVMs"
                    hbvm = RK(k=k, s=s, quadrature=quadrature, numeric=bool(k//4))
                    hbvm.write_to_tex(path="master-code/processed_data/butcher_tables", filename=tex_filename, mode="a")
                    hbvm.store_as_npz(path="master-code/src/butcher_tables")
        print("Done storing.")