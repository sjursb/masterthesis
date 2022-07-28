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
# For calculating gauss and lobatto weights quickly
import sympy.integrals.quadrature as q

# Make method class


class RK(param.Parameterized):
    """Class for storing Butcher table of an arbitrary Runge-Kutta method.

    The A, b and c coefficients are stored as numpy arrays of either numpy or sympy numbers (for compuation and visualization, respectively).

    The constructors use `sympy` CAS to calculate coefficients of HBVMs and collocation methods based on Gauss or Lobatto quadrature.

    Methods `hbvm`, `hbvm_gauss`, `hbvm_lobatto`, `lobattoIIIa` and `collocation` can be used for storing coefficients of RK methods, with `hbvm` being the most flexible (and also used in the class initialization). 
    """
    method_name = param.String(default="", doc="Name of RK-method")
    A = param.Array(doc='Bucher table A coefficient matrix')
    b = param.Array(doc='Butcher table b coefficients of method')
    c = param.Array(doc='Butcher table c coefficients (abscissae) of method')

    # For constructing method from HBVM or collocation framework
    s = param.Integer(doc='Rank of matrix A')
    k = param.Integer(doc='Number of stages in RK method')
    quadrature = param.Selector(
        default=0, objects=[0, 3], doc='Quadrature basis of method: 0=Gauss, 3=Lobatto')
    numeric = param.Boolean(
        default=True, doc="Info about whether the Butcher table is stored as numeric or symbolic objects (using either np.float or sympy objects).")

    def __init__(self, **kwargs):
        """Constructs Butcher table class instance from input parameters.

        If no name is supplied, 
        If no coefficients A, b, c 

        """
        super().__init__(**kwargs)
        if self.k < self.s:
            self.k = self.s
        if self.quadrature != 0 and self.quadrature != 3:
            self.quadrature = 3
        self.set_name(self.method_name)
        if self.A is None:
            if self.load_from_npz():
                pass
            else:
                self.hbvm(self.k, self.s, numeric=self.numeric,
                          quadrature=self.quadrature)

    @param.depends("k", "s", "quadrature")
    def set_name(self, name="", override_name=False):
        """
        set_name updates method_name either with input name or, if override_name is True, based on parameters k, s and quadrature after quadrature-HBVM(k, s) convention used in Master's Thesis.

        Parameters
        ----------
        name : str, optional
            New name of method, by default ""
        override_name : bool, optional
            Whether to override previous method_name value, by default False
        """
        if name:
            self.method_name = name
        elif override_name or not self.method_name:
            method_dict = {0: "Gauss", 3: "Lobatto"}
            if self.k == self.s:
                self.method_name = r"{}-{} or HBVM({},{})".format(
                    method_dict[self.quadrature], 2*self.k, self.k, self.k)
            else:
                self.method_name = r"{}-HBVM({},{})".format(
                    method_dict[self.quadrature], self.k, self.s)
        else:
            print("Name already set - supply name or override_name=True to change!")

    def load_from_npz(self, filename: str = None, path="."):
        """
        load_from_npz loads arrays A, b, and c from `filename` (.npz file) into class.

        If no `filename` is supplied, the method_name parameter and `path` keyword argument is used to construct filename.

        Returns True if load is successful or False otherwise.

        Parameters
        ----------
        filename : str, optional
            name (with relative path and file ending) of file with relevant Butcher table coefficients, by default None
        path : str, optional
            relative path of directory containing the .npz file with coefficient arrays, by default "."

        """

        if filename is None:
            filename = f"{path}\{self.method_name}.npz"
        if os.path.exists(filename):
            data = np.load(filename)
            self.A, self.b, self.c = data['A'], data['b'], data['c']
            data.close()
            return True
        else:
            return False

    def hbvm_lobatto(self, k, s, numeric=True):
        """Build Hamiltonian Boundary Value method with k + 1 stages and coefficient matrix of rank s using Lobatto quadrature."""
        self.k, self.s, self.A, self.b, self.c = k, s, * \
            get_Abc(k, s, numeric, quadrature=3)

    def hbvm_gauss(self, k, s, numeric=True):
        """Build Hamiltonian Boundary Value method with k stages and coefficient matrix of rank s using Gauss quadrature."""
        self.k, self.s, self.A, self.b, self.c = k, s, * \
            get_Abc(k, s, numeric, quadrature=0)

    def hbvm(self, k, s, numeric=True, quadrature:int=None):
        """
        hbvm constructs and stores an arbitrary HBVM(k, s) based on supplied quadrature in class instance.

        It uses hbvm_lobatto and hbvm_gauss as subroutines.

        Parameters
        ----------
        k : int
            number of (explicit) stages of the method
        s : int
            number of fundamental stages of the method / rank of A
        numeric : bool, optional
            whether to store coefficients as `numpy` (True) or `sympy` (False) numbers, by default True
        quadrature : int, optional
            quadrature type on which the method is based , by default None
        """
        if quadrature is None:
            quadrature = self.quadrature
        if quadrature == 0:
            self.hbvm_gauss(k, s, numeric=numeric)
        elif quadrature == 3:
            self.hbvm_lobatto(k, s, numeric=numeric)
        else:
            self.hbvm_lobatto(k, s, numeric=numeric)

    def lobattoIIIa(self, s, numeric=True):
        """Build Lobatto IIIa method with s stages (uses hbvm method internally).
        
        Parameters
        ----------
        s : int
            number of explicit stages
        numeric : bool, optional
            whether to store coefficients as `numpy` (True) or `sympy` (False) numbers, by default True
        """
        self.hbvm(s, s, numeric=numeric)

    def collocation(self, s, quadrature=None, numeric=None):
        """Build Butcher table for collocation method of order 2s with type 0 or III quadrature.
        
        Unless supplied, quadrature and numeric is taken from the class instance parameter values.

        Parameters
        ----------
        s : int
            number of (implicit) stages of the method
        quadrature : int, optional
            quadrature type on which the method is based (0 for Gauss and 3 for Lobatto), by default None
        numeric : bool, optional
            whether to store coefficients as `numpy` (True) or `sympy` (False) numbers, by default None 
        """
        if numeric is None:
            numeric = self.numeric
        if quadrature is None:
            quadrature = self.quadrature

        self.k, self.s = s, s
        self.A = collocation_A(s, numeric, quadrature=quadrature)
        self.b = collocation_b(s, numeric=numeric, quadrature=quadrature)
        self.c = get_c(s, numeric=numeric, quadrature=quadrature)

    def is_same_method(self, other_method):
        """
        Check if self instance and other instance of same class are same method (both are assumed numeric)

        Parameters
        ----------
        other_method : RK
            other instance of RK class to be compared with self instance
        
        Return 
        ------
        bool
            True if same method, False otherwise
        """
        return (np.allclose(self.A, other_method.A) and np.allclose(self.b, other_method.b) and np.allclose(self.c, other_method.c))

    def write_to_tex(self, filename:str= None, path=".", numeric=True, mode="w", verbose=True):
        """Write Butcher table to tex file. NB! Overwrites old file if it exists by default, so be careful.
        
        Parameters
        ----------
        filename : str, optional
            name of file to which the tables are stored, by default None
        path : str, optional
            relative path of directory where the file is to be stored as string, by default "."
        numeric : bool, optional
            whether to store as float (True) or expression (False), by default True
        mode : str, optional
            mode of writing to file ("w" overwrites old files), by default "w"
        verbose : bool, optional
            whether to print helpful statements in terminal during execution, by default True

        """
        if filename is None:
            filename = f"{self.method_name}.tex"
        elif filename[-4:] != ".tex":
            filename += ".tex"
        else:
            pass
        filename = f"{path}/{filename}"

        A, b, c = self.A, self.b, self.c
        latex_output = [latex(Matrix(A)), latex(Matrix(b).T), latex(Matrix(c))]
        with open(filename, mode) as texfile:
            texfile.write('\n' + self.method_name + '\n')
            for line, variable in zip(latex_output, ['A = ', 'b = ', 'c = ']):
                texfile.write(variable + line + '\n')

        if verbose:
            print("Butcher table written to file '{}'".format(filename))

    def store_as_npz(self, filename: str = None, path=".", overwrite=False, verbose=True):
        """Store Butcher table A, b, c in `filename.npz`, with option to overwrite existing files (default False).
        
        If no filename is supplied, the filename is constructed from method_name parameter.

        Parameters
        ----------
        filename : str, optional
            (optional) name of file in which the coefficients of the method is stored, by default None
        path : str, optional
            (relative) path to directory where the file is stored, by default "."
        overwrite : bool, optional
            whether to overwrite existing files, by default False
        verbose : bool, optional
            whether to print helpful statements in terminal during execution, by default True

        """
        if filename is None:
            filename = f"{self.method_name}"
        filename = f"{path}/{filename}"
        if os.path.exists(filename+".npz") and overwrite is False:
            if verbose:
                print(f"The method is already stored in file {filename}!")
        else:
            np.savez_compressed(filename, A=self.A, b=self.b, c=self.c)
            if verbose:
                print("Butcher table stored in file '{}'".format(filename))
            else: 
                pass

def shifted_legendre(n: int, x=symbols('x'), sympy=True):
    """shifted_legendre returns shifted (but not normalized!) Legendre polynomials of order n

    Parameters
    ----------
    n : int
        order of the Legendre polynomial
    x : sympy.core.symbol.Symbol, optional
        symbol used in polynomial, by default symbols('x')
    sympy : bool, optional
        whether to use built-in legendre method for construction, by default True

    """
    if sympy:
        return legendre(n, 2 * x - 1)
    if n == 0:
        result = (1)
    elif n == 1:
        result = (2 * x - 1)
    else:
        result = (((2 * n - 1) * (2 * x - 1) * shifted_legendre(n -
                  1, x) - (n - 1) * shifted_legendre(n - 2, x)) / n)
    return result


def lagrange(i: int, c: np.ndarray, x=symbols('x')):
    """lagrange constructs Lagrange polynomial `sympy` expression based on root vector c, index i and symbol x

    Parameters
    ----------
    i : int
        index of root upon which the polynomial is based
    c : np.ndarray
        array of roots of the polynomial
    x : sympy.core.symbol.Symbol
        `sympy` variable used in , by default symbols('x')
    """
    if type(c) == list:
        c = np.array(c)
    nominator = prod(x - c[c != c[i]])
    denominator = prod(c[i] - c[c != c[i]])
    return expand(nominator/denominator)


def get_roots(n, quadrature: int = 3) -> List[CRootOf]:
    """
    Get roots of Gauss or Lobatto polynomials of degree n using `sympy` CAS routines.

    Identical output to fsolve(f(x) = 0, x) in Maple for respective polynomials.

    Parameters
    ----------
    n : int
        degree of the relevant polynomial
    quadrature : int, optional
        Polynomial type for which roots are found (0 for Gauss and 3 for Lobatto), by default 3 
    """
    if quadrature == 0:  # Gauss quadrature
        f: Poly = Poly(shifted_legendre(n, x), x)

    elif quadrature == 3:  # Lobatto quadrature
        f: Poly = Poly((x**2 - x) * diff(shifted_legendre(n, x)), x)
    else:
        Exception()
    # root_dict = polys.polyroots.roots(f, x)
    return f.all_roots()


# Alternative function definitions
def gauss_roots(n): return get_roots(n, quadrature=0)
def lobatto_roots(n): return get_roots(n, quadrature=3)


def get_c(k: int, numeric=False, quadrature: int = 3):
    """
    Gets c coefficient array for Gauss or Lobatto polynomials from roots of the relevant polynomial.
    ### Note:
    - Lobatto has k + 1 stages (the first equal to the old value), whereas Gauss has k
        - This ensures that both quadrature have order 2k
        - However, c for Lobatto is of length k+1, whereas Gauss c has length k

    Parameters
    ----------
    k : int
        number of (implicit) stages of the method
    numeric : bool, optional
        whether to return coefficients as `numpy` (True) or `sympy` (False) numbers, by default False
    quadrature : int, optional
        the type of polynomial used to construct the method (0 for Gauss and 3 for Lobatto), by default 3
    """
    rts: List[CRootOf] = get_roots(
        k, quadrature=quadrature)  # False returns Lobatto nodes
    if numeric:
        return np.array([rt.n() for rt in rts]).astype(np.float64)
    else:
        return np.array(rts)


def gauss_b(k, c: "np.ndarray|None" = None, numeric: bool = False) -> np.ndarray:
    """
    Calculate b coefficients in Butcher table by formula 2.61 in report (for Gauss nodes)

    Note: This construction is significantly faster than the more general collocation_b.

    Parameters
    ----------
    k : int
        number of (implicit) stages of the method
    c : np.ndarray|None, optional
        pre-calculated abscissae of the method, by default None
    numeric : bool, optional
        whether to return coefficients as `numpy` (True) or `sympy` (False) numbers, by default False
    """

    if c is None:
        c = get_c(k, quadrature=0)
    P = diff(shifted_legendre(k, x), x)
    b = [simplify(1 / (c_i * (1 - c_i) * P.subs(x, c_i)**2)) for c_i in c]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)


def lobatto_b(k, c: "np.ndarray|None" = None, numeric: bool = False) -> np.ndarray:
    """
    Calculate b coefficients in Butcher table by formula 2.61 in report (for Lobatto nodes)

    Note: This construction is significantly faster than the more general collocation_b.

    Parameters
    ----------
    k : int
        number of (implicit) stages of the method
    c : np.ndarray|None, optional
        pre-calculated abscissae of the method, by default None
    numeric : bool, optional
        whether to return coefficients as `numpy` (True) or `sympy` (False) numbers, by default False
    """
    if c is None:
        c = get_c(k, numeric=numeric, quadrature=3)
    b: List[Rational] = [1 / (k*(k+1)*shifted_legendre(k, c[i])**2)
                         for i in range(k+1)]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)


def collocation_b(k, c: "np.ndarray|None" = None, numeric: bool = False, quadrature=0) -> np.ndarray:
    """
    Calculates b from B(s) condition, which is solved as a linear system.

    ### Notes: 
    - Only works for Gauss and Lobatto quadrature
    - Lobatto has k + 1 stages, whereas Gauss has k
    - Redundant and pretty slow

    Parameters
    ----------
    k : int
        number of (implicit) stages of the method
    c : np.ndarray|None, optional
        pre-calculated abscissae of the method, by default None
    numeric : bool, optional
        whether to return coefficients as `numpy` (True) or `sympy` (False) numbers, by default False
    quadrature : int, optional
        type of quadrature basis for the method (0 is Gauss and 3 is Lobatto), by default 0
    """

    if c is None:
        c = get_c(k, quadrature=quadrature)
    if quadrature == 3:
        k += 1  # Adjust for Lobatto IIIA having more stages

    V = Matrix([[c[j]**i for j in range(k)] for i in range(k)])
    a = Matrix([[Rational(1, q)] for q in range(1, k+1)])
    # linsolve returns the solution as a Tuple within a FiniteSet, which must be unpacked and made a list - yeah, it's weird...
    b = list(linsolve((V, a)).args[0])
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)


def lagrange_b(k: int, c: "np.ndarray|None" = None, numeric=False, quadrature=0) -> np.ndarray:
    """Construction of b coefficients calculated by integrating the Lagrange polynomials based on c vector and given quadrature.

    Untested and unused, but can potentially work for other types of collocation polynomials and/or quadrature rules.
    """
    if c is None:
        c = get_c(k, quadrature=quadrature)
    b = [integrate(lagrange(i, c, x), (x, 0, 1)) for i in range(len(c))]
    if numeric:
        return np.array([b_i.n() for b_i in b]).astype(np.float64)
    else:
        return np.array(b)


def get_b(k: int, c: "np.ndarray|None" = None, numeric=False, quadrature=0) -> np.ndarray:
    """
    get_b calculates and returns b coefficient array of quadrature rule of order 2k of type specified by quadrature keyword.

    If numeric is True, fast_quadrature function is used for calculations.
    Otherwise, the construction is based on the supplied k and optional c vector and 

    Parameters
    ----------
    k : int
        number of implicit stages of method / half the order of the quadrature rule
    c : np.ndarray | None, optional
        array of c coefficients for method, by default None
    numeric : bool, optional
        whether the resulting coefficients are `numpy` (True) or `sympy` (False) numbers, by default False, by default False
    quadrature : int, optional
        Type of quadrature underlying the method, by default 0

    Returns
    -------
    b : np.ndarray
        b coefficients of RK method
    """
    if (quadrature == 0 or quadrature == 3) and numeric:
        b, _ = fast_quadrature(k, quadrature)
    elif quadrature == 0:
        return gauss_b(k, c, numeric)
    elif quadrature == 3:
        return lobatto_b(k, c, numeric)
    else:
        return collocation_b(k, c, numeric, quadrature)


def collocation_A(s, numeric: bool = False, quadrature=0):
    """Build collocation method A coefficient matrix of rank s and declare whether numeric or symbolic

    Construction by integral of Lagrange polynomial (see Geometric Numerical Integration, pp.31).

    Note: Redundant (setting k=s in hbvm functions gives the same methods).

    Parameters
    ----------
    s : int
        number of (implicit) stages of collocation method
    numberic : bool, optional
        whether the resulting coefficients are `numpy` (True) or `sympy` (False) numbers, by default False
    quadrature : int, optional
        Type of collocation polynomial (0 for Gauss and 3 for Legendre), by default 0

    """
    c = get_c(s, quadrature=quadrature, numeric=numeric)
    k = len(c)  # s for gauss(type 0), s+1 for lobatto(type 3)
    def l(i, x): return lagrange(i, c, x)
    A = Matrix(k, k, lambda i, j: expand(integrate(l(j, x), (x, 0, c[i]))))

    if numeric:
        A = np.array([[A[i, j].n() for j in range(k)]
                     for i in range(k)]).astype(np.float64)
    return np.array(A)


def fast_quadrature(k, quadrature=0):
    """Returns numeric b, c vectors (quadrature rule of order 2k) constructed using built-in sympy routines. (Much faster than other implementation!)"""
    if quadrature == 0:
        c, b = q.gauss_legendre(k, 16)
    elif quadrature == 3:
        c, b = q.gauss_lobatto(k + 1, 16)
    c: np.ndarray = (np.array(c, dtype=np.float64)+1)/2
    b: np.ndarray = np.array(b, dtype=np.float64)/2
    return b, c


def hbvm_A(k: int, s: int, numeric: bool = False, quadrature=0, b: np.ndarray = None, c: np.ndarray = None):
    """
    hbvm_A constructs coefficient matrix A for HBVM(k, s) based on given quadrature

    If b and c coefficients (the quadrature rule of order 2k) are supplied, these are used in the computations; otherwise, they are constructed as a subroutine.

    setting quadrature=0 returns coefficients based on Gauss quadrature, while quadrature=3 returns coefficients based on Lobatto quadrature

    Parameters
    ----------
    k : int
        number of (implicit) total stages of the method (Lobatto methods have one explicit stage)
    s : int
        number of fundamental stages / rank of resulting coefficient matrix / 
    numeric : bool, optional
        whether the resulting coefficients are `numpy` (True) or `sympy` (False) numbers, by default False
    quadrature : int, optional
        quadrature type used as basis of the method, by default 0
    b : np.ndarray, optional
        coefficient vector b of the method, by default None
    c : np.ndarray, optional
        coefficient vector c of the method, by default None

    Returns
    -------
    A : np.ndarray
        A coefficient matrix of HBVM IRK method
    """
    if b is None or c is None:
        if numeric:
            b, c = fast_quadrature(k, quadrature=quadrature)
        else:
            # Should not use numeric here for precision
            c = get_c(k, numeric=numeric, quadrature=quadrature)
            # Should not use numeric here for precision
            b = get_b(k, c=c, numeric=numeric, quadrature=quadrature)
    else:
        pass

    k = len(c)  # Adjusts for lobatto having k + 1 stages
    B = diag(b.tolist(), unpack=True)
    # scaling matrix - ensures normalization
    D = diag([(2*j + 1) for j in range(s)], unpack=True)
    I_bar = Matrix(k, s, lambda i, j: integrate(
        shifted_legendre(j, x), (x, 0, c[i])))
    W = Matrix(k, s, lambda i, j: shifted_legendre(j, c[i]))

    A = expand(I_bar @ D @ W.T @ B)

    if numeric:
        A2 = [[A[i, j].n() for j in range(k)] for i in range(k)]
        return np.array(A2).astype(np.float64)
    else:
        return np.array(A)


def hbvm_A_lobatto(k, s, numeric): return hbvm_A(
    k, s, numeric, quadrature=3)  # Lobatto-HBVM(k, s) A coefficients shorthand


def hbvm_A_gauss(k, s, numeric): return hbvm_A(
    k, s, numeric, quadrature=0)  # Gauss-HBVM(k, s) A coefficients shorthand


def get_Abc(k: int, s: int, numeric=True, quadrature=0):
    """
    get_Abc calculates and returns coefficient arrays A, b and c for HBVM(k, s) based on given quadrature

    For k=s, the standard s-stage collocation method (Gauss or Lobatto IIIA) is returned.

    For methods with few stages, the coefficients can optionally be returned as numpy numbers (numeric=False), which allow for nice printing.

    Parameters
    ----------
    k : int
        number of total stages or length of b array (for Lobatto methods, b is of length k+1)
    s : int
        number of fundamental stages / rank of A
    numeric : bool, optional
        whether the coefficients should be `numpy` (True) or `sympy` (False) numbers, by default True
    quadrature : int, optional
        quadrature basis of method, by default 0

    Returns
    -------
        A, b and c coefficients as np.ndarrays
    """
    if numeric and (quadrature == 0 or quadrature == 3):
        b, c = fast_quadrature(k, quadrature)
    else:
        if k > 3 and quadrature == 0 and not numeric:
            numeric_c = True
        if k > 4 and quadrature == 3 and not numeric:
            numeric_c = True
        else:
            numeric_c = numeric
        c = get_c(k, numeric_c, quadrature)
        b = get_b(k, c, numeric, quadrature)
    A = hbvm_A(k, s, numeric, quadrature, b, c)

    return A, b, c


def plot_time_get_b(k_max=10, numeric=True, symbolic=False):
    """
    plot_time_get_b Generates plot showing time spent constructing b vector using functions `get_b` or `lobatto_b`

    Parameters
    ----------
    k_max : int, optional
        maximum number of stages, by default 10
    numeric : bool, optional
        whether to compare `numpy` construction, by default True
    symbolic : bool, optional
        whether to compare `sympy` symbolic number construction (very slow), by default False
    """
    ks = np.arange(1, k_max+1)
    js = []
    if symbolic:
        js += [0, 1]
    if numeric:
        js += [2, 3]
    times = np.zeros((len(js), k_max))
    for i in range(len(js)):
        for k in ks:
            t0 = time.time()
            if js[i] % 2:
                lobatto_b(k, numeric=js[i]//2)
            else:
                collocation_b(k, numeric=js[i]//2, quadrature=3)
            times[i, k-1] = time.time() - t0

    plt.figure()
    plt.title("Lobatto b vs general b")
    for i in range(len(js)):
        label = ""
        if js[i] % 2:
            label += "lobatto_b"
        else:
            label += "get_b"
        label += ", numeric={}".format(bool(js[i]//2))
        plt.plot(ks, times[i], label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if False:
        s = 2
        k = s  # k=s leads to collocation method
        quadrature = 0
    if False:
        # store all the necessary hbvm tables as .tex and .npz files
        quadratures = [0, 3]
        s_range = range(1, 5+1)

        quad_dict = {0: "Gauss", 3: "Lobatto"}
        for quadrature in quadratures:
            for s in s_range:
                silent_max = 3*s  # nu * s / 2, nu=6
                for r in range(silent_max+1):
                    k = s + r
                    tex_filename = f"{quad_dict[quadrature]}-HBVMs"
                    hbvm = RK(k=k, s=s, quadrature=quadrature,
                              numeric=bool(k//4))
                    hbvm.write_to_tex(
                        path="master-code/processed_data/butcher_tables", filename=tex_filename, mode="a")
                    hbvm.store_as_npz(path="master-code/src/butcher_tables")
        print("Done storing.")
