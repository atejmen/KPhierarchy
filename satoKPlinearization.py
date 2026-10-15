r'''
KP equation of flow M

    L = D + u1 D^-1 + u2 D^-2 + uM D^-M,        D = d/dx1
    d_{x_k} L = [B_k, L],                       B_k = (L^k)_+      (k = 2, M)

Run with SageMath:

    sage satoKPlinearization.py M

Steps: Lax equations -> linear triangularization with the independent terms carried along
-> (section) 7 non linear step that eliminates u2, ..., uM when possible.
'''
import os, sys

DALGEBRA_PARENT = "/home/sage/dalgebra-develop"
sys.path.insert(0, DALGEBRA_PARENT)

from sage.all import QQ, ZZ, PolynomialRing, Matrix, binomial
from dalgebra import DifferentialRing, DifferentialPolynomialRing

# ==== INPUT PARAMETER: M (KP equation number) ====
# Usage: sage satoKPlinearization.py M (M>=3; by default 3)

try:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if M < 3:
        raise ValueError
except ValueError:
    sys.exit("Usage: sage satoKPlinearization.py M (M integer >= 3)")
FLOWS = [2, M]  # flows x2 (B2), xM (BM)
TIMES = [1] + FLOWS

# ----------------------------------------------------------------------------
# 1. Differential Ring with TIMES derivatives and d-polynomials Ring
# ----------------------------------------------------------------------------
P = PolynomialRing(QQ, names=[f'x{t}' for t in TIMES])
R = DifferentialRing(P, *P.derivation_module().gens())   # d/dx1, d/dx_k, ...
DPR = DifferentialPolynomialRing(R, *[f'u{j}' for j in range(1, M +1)])

u = [g[(0,) * len(TIMES)] for g in DPR.gens()]   # d-polynomials u1, ..., uM (order 0 in all derivatives)
ZERO = DPR.zero()

# ----------------------------------------------------------------------------
# 2. Differential Pseudo-operators  sum_n a_n D^n  (n integer, D = d/dx1)
# ----------------------------------------------------------------------------
class PseudoDiffOp:
    def __init__(self, terms=None):
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    def multiply(self, other, min_order):
        r'''
        Product  (a D^i)(b D^j) = sum_{k>=0} binom(i, k) a b^(k) D^(i+j-k),
        with generalized binomial (valid for i < 0). Terms with
        order < min_order are discarded; the ones with order >= min_order are exact.
        '''
        res = {}
        for i, a in self.terms.items():
            for j, b in other.terms.items():
                jmax = i + j - min_order
                if i >= 0:                       # binom(i, k) = 0 if k > i
                    jmax = min(jmax, i)
                for k in range(jmax + 1):
                    term = DPR(ZZ(binomial(i, k))) * a * b.derivative(0, times=k)
                    res[i + j - k] = res.get(i + j - k, ZERO) + term
        return PseudoDiffOp(res)

    def positive_part(self):
        return PseudoDiffOp({k: v for k, v in self.terms.items() if k >= 0})

    def __sub__(self, other):
        res = dict(self.terms)
        for k, v in other.terms.items():
            res[k] = res.get(k, ZERO) - v
        return PseudoDiffOp(res)

def power(L, k):
    r'''L^k. It's truncated below -M so that (L^k)_+ is exact.'''
    result = L
    for _ in range(k - 1):
        result = result.multiply(L, -(M + k))
    return result

# ----------------------------------------------------------------------------
# 3. Lax equations
# ----------------------------------------------------------------------------
L = PseudoDiffOp({1: DPR.one(), **{-(j + 1): u[j] for j in range(M)}})

def lax_equations(L):
    r'''
    Equations  D_k u_j = D^-j coefficient in [B_k, L].
    As L is truncated in u_M, the D^-j coefficient in [B_k, L] only is
    exact if j < M + 2 - k (otherwise, u_{M+1} would be missing). For M = 3 the resulting terms are
    D2 u1, D2 u2 y D3 u1.
    '''
    eqs = []
    for k in FLOWS:
        Bk = power(L, k).positive_part()
        bracket = Bk.multiply(L, -M) - L.multiply(Bk, -M)
        assert all(order < 0 for order in bracket.terms), "[B_k, L] must have only negative order"
        for j in range(1, M + 2 - k):
            eqs.append(u[j - 1].derivative(TIMES.index(k)) - bracket.terms.get(-j, ZERO))
    return eqs

pdes = lax_equations(L)

# ----------------------------------------------------------------------------
# 4. Linearization in u = 0: matrix over QQ[D1, ..., DM]
# ----------------------------------------------------------------------------
S = PolynomialRing(QQ, names=[f'd{t}' for t in TIMES])

def _to_QQ(c):
    try:
        return QQ(getattr(c, 'wrapped', c))
    except (TypeError, ValueError):
        raise ValueError(f"Coefficient not constant in linear part: {c}")

def _linear_terms(eq):
    r'''
    Iterating over (generator index, order tuple, coefficient) of the
    monomials of degree 1 of ``eq``.
    '''
    for v in eq.variables():
        mon = v.monomials()[0]
        (gen_index, orders), _ = next(iter(mon._variables.items()))
        coeff = eq.coefficient_full(mon).constant_coefficient()
        if coeff != 0:
            yield gen_index, orders, coeff
    return

def linear_matrix(pdes, S):
    r'''
    Row i, column j: operator that multiplies u_j in linear part of pdes[i].
    The monomials of degree 1 are variables u_j[a,b,...] = D1^a D2^b ... DM^z u_j.
    '''
    ds = S.gens()
    A = [[S.zero() for _ in range(M)] for _ in pdes]
    for i, eq in enumerate(pdes):
        for gen_index, orders, coeff in _linear_terms(eq):
            orders = tuple(orders) if isinstance(orders, (tuple, list)) else (orders,)
            op = S.one()
            for d, e in zip(ds, orders):
                op *= d**e
            A[i][int(gen_index)] += _to_QQ(coeff) * op
    return Matrix(S, A)

A = linear_matrix(pdes, S)

def apply_op(op, f):
    r'''Operator op (polynomial in d1,d2,d3) is applied to d-polynomial f.'''
    res = ZERO
    for expo, c in op.dict().items():
        g = f
        for k, n in enumerate(expo):
            if n:
                g = g.derivative(k, times=n)
        res += DPR(c) * g
    return res

# Each equation is  (linear part) + (non linear part) = 0, i.e. A·u = b with
# b = -(non linear part): the "independent term" of the system.
b = []
for i, eq in enumerate(pdes):
    lineal = sum((apply_op(A[i, j], u[j]) for j in range(M)), ZERO)
    b.append(lineal - eq)

# ----------------------------------------------------------------------------
# 5. GCD elimination (conmutative ring: coefficients are constant)
# ----------------------------------------------------------------------------
def ore_triangularize_general(A,b):
    r'''
    Eliminates column to column from last of the system A·u = b 
    (b: d-polynomial independent terms). For each column it is choosen as
    pivot the free row of minimum degree and, for each other free row r with not null
    input, it is done row_r < - (a/g) row_r - (b_/g) row_p,  with a = A[p,col],
    b_ = A[r,col], g = gcd(a,b) and the same combination, aplying the operators to
    the d-polynomials indepedent terms. Each new row ia a consequence of the previous ones.

    Each row is not divided by its content (inputs GCD): P*u1 = 0 is not
    equivalent to u1 = 0 when coefficients are operators.

    It returns (T,c): T is the triangular matrix and c is
    the transformed indepedent terms, such that T·u = c
    '''
    S = A.base_ring()
    nr, nc = A.nrows(), A.ncols()
    nA = [list(A.row(i)) for i in range(nr)]
    rhs = list(b)
    used, pivot_of = set(), {}
    
    for col in reversed(range(nc)):
        cand = [r for r in range(nr) if r not in used and nA[r][col] != 0]
        if not cand:
            continue
        p = min(cand, key=lambda r: nA[r][col].total_degree())
        used.add(p)
        pivot_of[col] = p
        for r in cand:
            if r == p:
                continue
            a, b_ = nA[p][col], nA[r][col]
            g = a.gcd(b_)
            ca, cb = a // g, b_ // g
            nA[r] = [ca * nA[r][j] - cb * nA[p][j] for j in range(nc)]
            rhs[r] = apply_op(ca, rhs[r]) - apply_op(cb, rhs[p])   
    order = [r for r in range(nr) if r not in used] + [pivot_of[c] for c in sorted(pivot_of)]
    return Matrix(S, [nA[r] for r in order]), [rhs[r] for r in order]
    
A_triangular, b_triangular = ore_triangularize_general(A, b)

# ----------------------------------------------------------------------------
# 6. Results
# ----------------------------------------------------------------------------
print("--- Lax equations ---")
for eq in pdes:
    print(eq, "= 0")

print("\n--- Linear Matrix (A) and independent terms ---")
print(A)
for i, d in enumerate(b):
    print(f"  b[{i}] =", d)

print("\n--- Triangularized Matrix (A_triangular) and independent terms ---")
print(A_triangular)
for i, d in enumerate(b_triangular):
    print(f"  b[{i}] =", d)

def fields_in(f):
    r'''Indexes (0 = u1) of the fields that appear in the d-polynomial f.'''
    return sorted({int(gi) for mon in f.monomials() for (gi, _) in mon._variables})

# Rows that involves just u1: T[i,0]·u1 = b[i]
rows_u1 = [i for i in range(A_triangular.nrows())
           if all(A_triangular[i, j] == 0 for j in range(1, M))]
equations_u1 = []
print(f"\n--- Equations for u1 (M={M}, flows={FLOWS}) ---)")
for i in rows_u1:
    eq = apply_op(A_triangular[i, 0], u[0]) - b_triangular[i]
    equations_u1.append(eq)
    extra = [f"u{j + 1}" for j in fields_in(b_triangular[i]) if j != 0]
    print(f"row {i}: linear part = {A_triangular[i,0].factor()}")
    print("   ", eq, "= 0")
    if extra:
        print(f"    WARNING: the independent term still contains {', '.join(extra)}; "
              "the equation is not closed in u1 (see final note)")

# NOTE: the elimination is linear (the coeffcients are constant operators).
# The non linear terms progress unalterated, so only are closed in u1 when
# they don't contain u2, ..., uM. In other case, u2, ..., uM have to be sustituted
# by expressions in u1 using D1^-1 (no local), for instance 2 D1 u2 = (D2 - D1^2) u1.

# ----------------------------------------------------------------------------
# 7. Non linear step: elimination of u2, ..., uM
# ----------------------------------------------------------------------------
# The equations of flow x2 are   D2 u_j = u_j'' + 2 u_{j+1}' + (terms in u1, ..., u_j),
# so   u_{j+1}' = W_{j+1}(u1, ..., u_j and their derivatives)   (local, no integrals).
# Therefore every variable u_j[a,b,...] with j >= 2 and a >= 1 (at least one x1-derivative)
# can be replaced by a polynomial in lower fields.  Only the u_j with no x1-derivative
# (a = 0) cannot be replaced.  If a single such variable z remains and it appears linearly,
#       E = c*z + N      (c, N free of z),     D1 z = dz  (known, by the same rule)
# then   F = D1(c)*N - c*D1(N) - c^2*dz = 0   is an equation in u1 only (no integrals):
#       D1(c)*E - c*D1(E)  =  D1(c)*N - c*(c*dz + D1(N)).
GENS = DPR.gens()

def _orders(o):
    return tuple(o) if isinstance(o, (tuple, list)) else (o,)

def _terms(f):
    r'''Iterates over (key, coefficient in QQ) of the monomials of f, where
    key = {(field index, order tuple): exponent}.'''
    for mon in f.monomials():
        key = {(int(gi), _orders(o)): int(e) for (gi, o), e in mon._variables.items()}
        if key:
            coeff = f.coefficient_full(mon).constant_coefficient()
        else:
            coeff = f.constant_coefficient()
        yield key, _to_QQ(coeff)

def _Dn(f, orders):
    r'''Applies D1^orders[0] D2^orders[1] ... to the d-polynomial f.'''
    for k, n in enumerate(orders):
        if n:
            f = f.derivative(k, times=n)
    return f

W = {}   # W[j] = D1 u_{j+1} (0-based field index j >= 1) as d-polynomial in lower fields

def W_of(j):
    if j not in W:
        Rj = pdes[j - 1] + 2 * u[j].derivative(0)    # = D2 u_j - u_j'' - (non linear terms)
        for key, _ in _terms(Rj):
            if any(gi == j and o[0] >= 1 for (gi, o) in key):
                raise RuntimeError(f"unexpected x1-derivative of u{j + 1} in the flow-x2 equation of u{j}")
        W[j] = reduce_x(DPR(QQ(1) / 2) * Rj)
    return W[j]

def reduce_x(f):
    r'''Replaces every u_j[a,b,..] (j >= 2, a >= 1) by D^(a-1,b,..) W_j, recursively.'''
    res = ZERO
    for key, coeff in _terms(f):
        term = DPR(coeff)
        for (gi, o), e in key.items():
            if gi >= 1 and o[0] >= 1:
                rep = reduce_x(_Dn(W_of(gi), (o[0] - 1,) + o[1:]))
            else:
                rep = GENS[gi][o]
            term *= rep ** e
        res += term
    return res

def free_variables(f):
    r'''Variables of the fields u2, ..., uM that remain in f.'''
    return {(gi, o) for key, _ in _terms(f) for (gi, o) in key if gi >= 1}

def eliminate_free_variable(E, zkey):
    r'''E = c*z + N  ->  F = D1(c)*N - c*D1(N) - c^2*D1(z). None if z is not linear in E.'''
    gz, oz = zkey
    c, Nrest = ZERO, ZERO
    for key, coeff in _terms(E):
        mon = DPR(coeff)
        for (gi, o), e in key.items():
            if (gi, o) != zkey:
                mon *= GENS[gi][o] ** e
        ez = key.get(zkey, 0)
        if ez > 1:
            return None
        if ez == 1:
            c += mon
        else:
            Nrest += mon
    if c == 0:
        return None
    dz = reduce_x(_Dn(W_of(gz), (0,) + oz[1:]))      # D1 z, with z = u_{gz+1}[0, b, ...]
    return c.derivative(0) * Nrest - c * Nrest.derivative(0) - c * c * dz

def _name(gi, o):
    return f"u{gi + 1}[{','.join(map(str, o))}]"

print("\n--- Elimination of u2, ..., uM (non linear step) ---")
final_equations = []
for eq in equations_u1:
    E = reduce_x(eq)
    free = free_variables(E)
    if not free:
        F = E
        print("Local equation in u1 (no integrals):")
    elif len(free) == 1:
        F = eliminate_free_variable(E, next(iter(free)))
        if F is None:
            print(f"Not closed: {_name(*next(iter(free)))} does not appear linearly.")
            print("   ", E, "= 0")
            continue
        print(f"Eliminated {_name(*next(iter(free)))}; local equation in u1 (no integrals):")
    else:
        print("Not closed: several variables without x1-derivative remain:",
              ", ".join(sorted(_name(*k) for k in free)))
        print("   ", E, "= 0")
        continue
    final_equations.append(F)
    print("   ", F, "= 0")

# NOTE: for M = 3 no u2, u3 remain after the linear step. For M = 4 only u2 (undifferentiated)
# remains and the step above removes it.  For M >= 5 several undifferentiated fields
# (u2, u3, and u2[0,1,0], ...) can remain at once; this step does not handle that case and
# the script just reports them.