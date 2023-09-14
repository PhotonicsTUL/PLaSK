import sys
import re

import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
from sympy import latex
from sympy.printing import cxxcode

try:
    from sympy.printing.cxx import CXX11CodePrinter
except ImportError:
    from sympy.printing.cxxcode import CXX11CodePrinter

from mpi4py.MPI import COMM_WORLD as mpi

msize = mpi.Get_size()
mrank = mpi.Get_rank()


def evaluate(expr, args):
    largs = np.array_split(args, msize)[mrank]
    if len(largs.shape) == 1:
        lres = [expr(arg) for arg in largs]
    else:
        lres = [expr(*arg) for arg in largs]
    res = mpi.gather(lres, root=0)
    if mrank == 0:
        return np.concatenate(res)


def evaluate_matrix(expr):
    args = [(i,j) for j in range(12) for i in range(j+1)]
    vals = evaluate(expr, args)
    if mrank == 0:
        matrix = sym.Matrix(np.zeros((12, 12)))
        for (i,j), val in zip(args, vals):
            matrix[i,j] = val
            if i != j:
                matrix[j,i] = val
        return matrix


def evaluate_vector(expr):
    args = range(12)
    vals = evaluate(expr, args)
    if mrank == 0:
        vector = sym.Matrix(np.zeros(12))
        for i, val in zip(args, vals):
            vector[i] = val
        return vector


def _print_Indexed(self, expr):
    indices = expr.indices
    if len(indices) == 1:
        # return f"{self._print(expr.base.label)}[idx(e,{self._print(indices[0])})]"
        return f"{self._print(expr.base.label)}{self._print(indices[0])}"
    else:
        # return f"{self._print(expr.base.label)}[idx(e,{','.join(self._print(i) for i in indices)})]"
        return f"{self._print(expr.base.label)}{''.join(self._print(i) for i in indices)}"
CXX11CodePrinter._print_Indexed = _print_Indexed


def cpp(v):
    s = cxxcode(sym.simplify(v), standard='C++11')
    #for i in range(1, 5):
    #    s = s.replace(f"a[{i}]", f"a{i}")
    s = re.sub(r'std::pow\(([^)]+), (\d+)\)', r'std::pow(\1,\2)', s)
    s = re.sub(r'std::pow\(([^)]+),2\)', r'(\1*\1)', s)
    s = re.sub(r'std::pow\(([^)]+),3\)', r'(\1*\1*\1)', s)
    s = re.sub(r'\b(X|Y)\b', r'e.\1', s)
    s = re.sub(r'\bU(\d\d)\b', r'U[e.i\1]', s)
    s = re.sub(r'\bJ(\d\d)\b', r'J[e.n\1]', s)
    s = re.sub(r'\bP(\d\d)(\d)\b', r'P[e.n\1].c\2\2', s)
    s = re.sub(r'(d?G)(\d)\b', r'\1.c\2\2', s)
    return s


sidx = lambda i: "e.i{}{}".format(*idx[i])

def _print_cpp(kij, kvals, fvals, file=sys.stdout):
    for (i,j), kval in zip(kij, kvals):
        print(f"K({sidx(i)}, {sidx(j)}) += {kval};", file=file)
    for i, fval in enumerate(fvals):
        print(f"F[{sidx(i)}] += {fval};", file=file)


def print_cpp(K, F, fname=None):
    kij = [(i,j) for j in range(12) for i in range(j+1)]
    kvals = evaluate(lambda i,j: cpp(K[i,j]), kij)
    fvals = evaluate(lambda i: cpp(F[i]), range(12))
    if mrank == 0:
        if fname is None:
            _print_cpp(kij, kvals, fvals)
        else:
            with open(fname, 'w') as file:
                _print_cpp(kij, kvals, fvals, file=file)


# Podstawowe obliczenia

A, B, C, D, = sym.symbols('A B C D')

x, y = sym.symbols('x y')
X, Y = sym.symbols('X Y')

ξ = x / X
ζ = y / Y

Φ = np.linalg.inv(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 2, 3]])).T

φx = [sum(int(c) * ξ**i for i, c in enumerate(p)) for p in Φ]
φx[1] *= X
φx[3] *= X
φx = np.array([sym.expand(a) for a in φx])

φy = [sum(int(c) * ζ**i for i, c in enumerate(p)) for p in Φ]
φy[1] *= Y
φy[3] *= Y
φy = np.array([sym.expand(a) for a in φy])

φ = np.array([[φx[0] * φy[0], φx[1] * φy[0], φx[2] * φy[0], φx[3] * φy[0]], [φx[0] * φy[1], 0, φx[2] * φy[1], 0],
              [φx[0] * φy[2], φx[1] * φy[2], φx[2] * φy[2], φx[3] * φy[2]], [φx[0] * φy[3], 0, φx[2] * φy[3], 0]])

idx = [(0,0), (0,1), (1,0), (0,2), (0,3), (1,2), (2,0), (2,1), (3,0), (2,2), (2,3), (3,2)]
φ = np.array([φ[idx[i]] for i in range(12)])

dφx = np.array([sym.expand(sym.diff(a, x)) for a in φ])
dφy = np.array([sym.expand(sym.diff(a, y)) for a in φ])

U = sym.IndexedBase('U', shape=(4, 4))
u0 = sum(U[idx[k]] * φ[k] for k in range(12))

zx = np.array([0, X])
zy = np.array([0, Y])

J = sym.IndexedBase('J', shape=(2, 2))


def La(i, x, z):
    res = 1
    for j in range(len(z)):
        if j == i: continue
        res *= (x - z[j]) / (z[i] - z[j])
    return res


Jxy = sym.simplify(sum(J[i,j] * La(i, x, zx) * La(j, y, zy) for i in range(len(zx)) for j in range(len(zy))))


KD = evaluate_matrix(lambda i,j:
    sym.simplify(sym.integrate(sym.integrate(D * (dφx[i] * dφx[j] + dφy[i] * dφy[j]), (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('KD = sym.', KD, end='\n\n')

KA = evaluate_matrix(lambda i,j:
    sym.simplify(sym.integrate(sym.integrate(A * φ[i] * φ[j], (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('KA = sym.', KA, end='\n\n')

KB = evaluate_matrix(lambda i,j:
    sym.simplify(sym.integrate(sym.integrate(2 * B * u0 * φ[i] * φ[j], (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('KB = sym.', KB, end='\n\n')

u02 = sym.expand(u0**2)
u03 = sym.expand(u0**3)

KC = evaluate_matrix(lambda i,j:
    sym.simplify(sym.integrate(sym.simplify(sym.integrate(3 * C * u02 * φ[i] * φ[j], (x, 0, X))), (y, 0, Y)))
)
if mrank == 0: print('KC = sym.', KC, end='\n\n')

if mrank == 0:
    K = KD + KA + KB + KC
    print('K = sym.', K, end='\n\n')
else:
    K = None
K = mpi.bcast(K, root=0)

FB = evaluate_vector(lambda i:
    sym.simplify(sym.integrate(sym.simplify(sym.integrate(B * u02 * φ[i], (x, 0, X))), (y, 0, Y)))
)
if mrank == 0: print('FB = sym.', FB, end='\n\n')

FC = evaluate_vector(lambda i:
    sym.simplify(sym.integrate(sym.simplify(sym.integrate(2 * C * u03 * φ[i], (x, 0, X))), (y, 0, Y)))
)
if mrank == 0: print('FC = sym.', FC, end='\n\n')

F0 = evaluate_vector(lambda i:
    sym.simplify(sym.integrate(sym.integrate(Jxy * φ[i], (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('F0 = sym.', F0, end='\n\n')

if mrank == 0:
    F = FB + FC + F0
    print('F = sym.', F, end='\n\n')
else:
    F = None
F = mpi.bcast(F, root=0)

print_cpp(K, F, 'diffusion3d-eval.ipp')


# Wypalanie nośników

P = sym.IndexedBase('P', shape=(2,2,2))
G = sym.IndexedBase('G', shape=(2,))
dG = sym.IndexedBase('dG', shape=(2,))
ug = sym.symbols('Ug')

Pxy = [sym.simplify(sum(P[i, j, c] * La(i, x, zx) * La(j, y, zy) for i in range(len(zx)) for j in range(len(zy)))) for c in range(2)]

KL = evaluate_matrix(lambda i,j:
    sym.simplify(sym.integrate(sym.integrate(sum(Pxy[i] * dG[i] for i in range(2)) * φ[i] * φ[j], (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('KL = sym.', KL, end='\n\n')

FL = evaluate_vector(lambda i:
    sym.simplify(sym.integrate(sym.integrate(sum(Pxy[i] * (dG[i] * ug - G[i]) for i in range(2)) * φ[i], (x, 0, X)), (y, 0, Y)))
)
if mrank == 0: print('FL = sym.', FL, end='\n\n')

KL = mpi.bcast(KL, root=0)
FL = mpi.bcast(FL, root=0)

print_cpp(KL, FL, 'diffusion3d-eval-shb.ipp')
