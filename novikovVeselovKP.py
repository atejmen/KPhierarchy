r'''
Bivariate (x, y) KP equation of flow M in the Novikov-Veselov / Zakharov-Shabat formulation,
and its expression in u1 only.

    L   = D + u1 D^-1 + ...,       D = d/dx,  x = t1,  y = t2
    N   = d_y - (L^2)_+  = d_y - (D^2 + 2 u1)
    B_M = (L^M)_+ = D^M + M u1 D^(M-2) + b_{M-3} D^(M-3) + ... + b_0     (b_j unknown functions)

    dN/dt_M = [B_M, N]          (zero curvature condition of the pair d_y - (L^2)_+, d_{t_M} - B_M;
                                 [N, B_M] reverses the sense of the time t_M)

The coefficients of D^(M-1), ..., D^0 give M equations in the M unknown functions
u1, b_0, ..., b_{M-3} of the three variables (x, y, t_M): the bivariate KP equation of flow M.

Then u1 alone is obtained as follows (the most efficient method among the ones that were tried:
Rosenfeld-Groebner and the elimination of dalgebra did not finish for M >= 4, while this one
solves M = 3, ..., 6):
  1. The equations of order D^(M-2), ..., D^1 give D b_j as local expressions in lower
     unknowns, so every x-derivative of b_j is replaced by a local expression.
  2. Only the b_j without x-derivative remain ("free variables", together with their y and
     t_M derivatives). The remaining equation E is polynomial in them; E_k = D^k E (k = 0, 1, ...)
     is linear in the monomials of the free variables, with coefficients in u1 only.
  3. The first k whose rows are linearly dependent (found with a random specialization modulo a
     prime) gives, as a determinant of the coefficients (fraction-free Bareiss algorithm in a
     Sage polynomial ring), an equation in u1 only.

Run with SageMath:

    sage novikovVeselovKP.py M

    M       integer >= 3                                           [by default 3]
'''
import os, sys, time, random

DALGEBRA_PARENT = "/home/sage/dalgebra-develop"
sys.path.insert(0, DALGEBRA_PARENT)

from sage.all import QQ, ZZ, PolynomialRing, Matrix, binomial
from dalgebra import DifferentialRing, DifferentialPolynomialRing

# ==== INPUT PARAMETER: M (flow number) ====
try:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if M < 3:
        raise ValueError
except ValueError:
    sys.exit("Usage: sage novikovVeselovKP.py M (M integer >= 3)")
TIMES = [1, 2, M]            # x = t1, y = t2, t_M
NDER = len(TIMES)

# ----------------------------------------------------------------------------
# 1. Differential ring in (x, y, t_M) and ring of d-polynomials
# ----------------------------------------------------------------------------
# Fields: index 0 is u1; index f >= 1 is b_{M-2-f}, i.e. b_{M-3}, b_{M-4}, ..., b_0.
P = PolynomialRing(QQ, names=['x', 'y', f't{M}'])
R = DifferentialRing(P, *P.derivation_module().gens())     # d/dx, d/dy, d/dt_M
FIELD_NAMES = ['u1'] + [f'b{j}' for j in range(M - 3, -1, -1)]
DPR = DifferentialPolynomialRing(R, *FIELD_NAMES)

# dalgebra SORTS the variable names (for M = 3 the order is b0, u1), so the position of a field in
# DPR.gens() is not its logical index.  PHYSICAL[f] is the position in DPR.gens() of the logical
# field f (0: u1, f >= 1: b_{M-2-f}) and LOGICAL is the inverse map.  The code below always uses
# logical indices; the translation is done only when generators are created (GENS[PHYSICAL[f]])
# and when the variables of a d-polynomial are read (_terms).
VNAMES = DPR.variable_names()
PHYSICAL = [VNAMES.index(name) for name in FIELD_NAMES]
LOGICAL = {p: f for f, p in enumerate(PHYSICAL)}

fields = [DPR.gens()[PHYSICAL[f]][(0,) * NDER] for f in range(len(FIELD_NAMES))]
u1 = fields[0]
ZERO = DPR.zero()


def b(j):
    r'''Coefficient b_j of D^j in B_M (0 <= j <= M-3).'''
    return fields[M - 2 - j]


# ----------------------------------------------------------------------------
# 2. Differential operators sum_n a_n D^n  (D = d/dx) and the zero curvature equations
# ----------------------------------------------------------------------------
class DiffOp:
    def __init__(self, terms=None):
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    def __mul__(self, other):
        r'''(a D^m)(c D^n) = sum_{j=0}^{m} binom(m, j) a c^(j) D^(m+n-j).'''
        res = {}
        for m, a in self.terms.items():
            for n, c in other.terms.items():
                for j in range(m + 1):
                    term = DPR(ZZ(binomial(m, j))) * a * c.derivative(0, times=j)
                    res[m + n - j] = res.get(m + n - j, ZERO) + term
        return DiffOp(res)

    def __sub__(self, other):
        res = dict(self.terms)
        for k, v in other.terms.items():
            res[k] = res.get(k, ZERO) - v
        return DiffOp(res)

    def derivative(self, k):
        r'''Derivative of the coefficients with respect to the k-th derivation (0: x, 1: y, 2: t_M).'''
        return DiffOp({n: c.derivative(k) for n, c in self.terms.items()})


B2 = DiffOp({2: DPR.one(), 0: 2 * u1})                                   # (L^2)_+
BM = DiffOp({M: DPR.one(), M - 2: M * u1, **{j: b(j) for j in range(M - 2)}})   # (L^M)_+
commutator = B2 * BM - BM * B2
BM_y = BM.derivative(1)

# 2 u1_{t_M} - (B_M)_y + [B_2, B_M] = 0, coefficient of D^k  (k = M-1, ..., 0)
zs = {}
for k in range(M - 1, -1, -1):
    e = commutator.terms.get(k, ZERO) - BM_y.terms.get(k, ZERO)
    if k == 0:
        e += 2 * u1.derivative(2)
    zs[k] = e
assert zs[M - 1] == 0, "the coefficient of D^(M-1) must vanish"

print(f"M={M}  times (x, y, t{M})  |  unknown functions: {', '.join(FIELD_NAMES)}")
print("--- Zero curvature equations  dN/dt_M = [B_M, N]  (coefficients of D^k) ---")
for k in range(M - 2, -1, -1):
    print(f"  D^{k}:  {zs[k]} = 0")

# ----------------------------------------------------------------------------
# 3. Translation helpers
# ----------------------------------------------------------------------------
GENS = DPR.gens()


def _to_QQ(c):
    try:
        return QQ(getattr(c, 'wrapped', c))
    except (TypeError, ValueError):
        raise ValueError(f"Non constant coefficient: {c}")


def _orders(o):
    return tuple(o) if isinstance(o, (tuple, list)) else (o,)


def _terms(f):
    r'''Iterates over (key, coefficient in QQ) of the monomials of the d-polynomial f, where
    key = {(logical field index, order tuple): exponent}.'''
    for mon in f.monomials():
        key = {(LOGICAL[int(gi)], _orders(o)): int(e) for (gi, o), e in mon._variables.items()}
        if key:
            coeff = f.coefficient_full(mon).constant_coefficient()
        else:
            coeff = f.constant_coefficient()
        yield key, _to_QQ(coeff)


def _Dn(f, orders):
    r'''Applies D_x^orders[0] D_y^orders[1] D_t^orders[2] to the d-polynomial f.'''
    for k, n in enumerate(orders):
        if n:
            f = f.derivative(k, times=n)
    return f


# ----------------------------------------------------------------------------
# 4. Replacement of the x-derivatives of b_j by local expressions
# ----------------------------------------------------------------------------
# The equation of D^(M-1-f) is  c * D b + (rest) = 0 with c a non zero constant, where b is the
# field f >= 1 (b_{M-2-f}); so  D b = -(rest) / c  is a local expression in u1 and in the fields
# of index < f (and their derivatives).
W = {}      # W[f] = D_x of the field f, as a d-polynomial in the fields of lower index
X_ORDER = (1,) + (0,) * (NDER - 1)


def W_of(f):
    if f not in W:
        E = zs[M - 1 - f]
        c = None
        for key, coeff in _terms(E):
            if key == {(f, X_ORDER): 1}:
                c = coeff
        if c is None or c == 0:
            raise RuntimeError(f"the equation of D^{M - 1 - f} does not contain D b{M - 2 - f}")
        rest = E - DPR(c) * GENS[PHYSICAL[f]][X_ORDER]
        for key, _ in _terms(rest):
            if any(gi == f and o[0] >= 1 for (gi, o) in key):
                raise RuntimeError("unexpected x-derivative of the unknown in the equation")
        W[f] = reduce_x(DPR(-1 / c) * rest)
    return W[f]


def reduce_x(f):
    r'''Replaces every variable of a field f >= 1 with x-order >= 1 by its local expression.'''
    res = ZERO
    for key, coeff in _terms(f):
        term = DPR(coeff)
        for (gi, o), e in key.items():
            if gi >= 1 and o[0] >= 1:
                rep = reduce_x(_Dn(W_of(gi), (o[0] - 1,) + o[1:]))
            else:
                rep = GENS[PHYSICAL[gi]][o]
            term *= rep ** e
        res += term
    return res


def free_variables(f):
    r'''Variables of the fields b_j that remain in f (all of them with x-order 0).'''
    return {(gi, o) for key, _ in _terms(f) for (gi, o) in key if gi >= 1}


# ----------------------------------------------------------------------------
# 5. Elimination of the free variables: determinant of the derivatives of the equation
# ----------------------------------------------------------------------------
PRIME = 2147483647


def _atom_split(f):
    r'''f = sum of coeff * atom, atom = monomial in the variables of the fields b_j (() = constant).
    Returns {atom: coeff}, the coefficients being d-polynomials in u1 only.'''
    res = {}
    for key, coeff in _terms(f):
        mon, atom = DPR(coeff), []
        for k, e in key.items():
            if k[0] >= 1:
                atom.append((k, e))
            else:
                mon *= GENS[PHYSICAL[k[0]]][k[1]] ** e
        atom = tuple(sorted(atom))
        res[atom] = res.get(atom, ZERO) + mon
    return res


def _eval_mod(f, point):
    r'''Value of the d-polynomial f (u1 only) modulo PRIME at a random point (kept in `point`).'''
    total = 0
    for key, coeff in _terms(f):
        val = int(coeff.numerator()) * pow(int(coeff.denominator()), -1, PRIME) % PRIME
        for k, e in key.items():
            if k not in point:
                point[k] = random.randrange(1, PRIME)
            val = val * pow(point[k], e, PRIME) % PRIME
        total = (total + val) % PRIME
    return total


def _rank_pivots(Mx):
    r'''Rank and pivot columns of an integer matrix modulo PRIME.'''
    Mx = [row[:] for row in Mx]
    rank, pivots = 0, []
    for c in range(len(Mx[0]) if Mx else 0):
        p = next((r for r in range(rank, len(Mx)) if Mx[r][c] % PRIME), None)
        if p is None:
            continue
        Mx[rank], Mx[p] = Mx[p], Mx[rank]
        inv = pow(Mx[rank][c], -1, PRIME)
        Mx[rank] = [x * inv % PRIME for x in Mx[rank]]
        for r in range(len(Mx)):
            if r != rank and Mx[r][c] % PRIME:
                f = Mx[r][c]
                Mx[r] = [(a - f * c2) % PRIME for a, c2 in zip(Mx[r], Mx[rank])]
        rank += 1
        pivots.append(c)
    return rank, pivots


def _bareiss(Mx):
    r'''Fraction-free determinant (Bareiss) of a square matrix (list of lists) over a polynomial ring.'''
    n = len(Mx)
    A = [row[:] for row in Mx]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if swap is None:
                return A[0][0] * 0
            A[k], A[swap] = A[swap], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def _determinant_equation(rows, atoms, piv, k):
    r'''det [ C_{0..k, piv} | N_{0..k} ] as a d-polynomial in u1 (computed in a Sage polynomial ring).'''
    cols = [atoms[c] for c in piv] + [()]
    entries = [[rows[i].get(a, ZERO) for a in cols] for i in range(k + 1)]
    keys = sorted({kk for row in entries for f in row for key, _ in _terms(f) for kk in key})
    Rp = PolynomialRing(QQ, names=[f"w{i}" for i in range(len(keys))])
    vmap = {kk: Rp.gen(i) for i, kk in enumerate(keys)}

    def to_poly(f):
        res = Rp.zero()
        for key, coeff in _terms(f):
            mon = Rp(coeff)
            for kk, e in key.items():
                mon *= vmap[kk] ** e
            res += mon
        return res

    table = [[to_poly(f) for f in row] for row in entries]
    try:
        D = _bareiss(table)
    except Exception:                        # fallback: generic determinant of Sage
        D = Matrix(Rp, table).det()
    if D == 0:
        return None
    F = ZERO
    for mon, c in zip(D.monomials(), D.coefficients()):
        term = DPR(c)
        for v in mon.variables():
            gi, o = keys[Rp.gens().index(v)]
            term *= GENS[PHYSICAL[gi]][o] ** int(mon.degree(v))
        F += term
    return F


def eliminate_free_variables(E, max_rows=60):
    r'''
    Equation in u1 only from E (reduced; polynomial in u1 and in the free variables).
    Every monomial in the free variables is an unknown; E_k = D_x^k E is linear in them.
    Returns (F, info) or (None, reason).
    '''
    rows, values, atoms, point = [], [], [], {}
    Ek = E
    for k in range(max_rows):
        if k > 0:
            Ek = reduce_x(Ek.derivative(0))
        split = _atom_split(Ek)
        rows.append(split)
        for a in split:
            if a != () and a not in atoms:
                atoms.append(a)
        values.append({a: _eval_mod(c, point) for a, c in split.items()})
        C = [[v.get(a, 0) for a in atoms] for v in values]
        rank, piv = _rank_pivots(C)
        if rank < k + 1:
            rank_aug, _ = _rank_pivots([row + [v.get((), 0)] for row, v in zip(C, values)])
            if rank_aug == rank:
                return None, "the dependency among the equations gives no information (0 = 0)"
            F = _determinant_equation(rows, atoms, piv, k)
            if F is None:
                return None, "the determinant vanishes (degenerate system)"
            return F, f"{len(atoms)} unknown(s), {k + 1} equations"
    return None, f"no dependency found with {max_rows} equations"


# ----------------------------------------------------------------------------
# 6. Run
# ----------------------------------------------------------------------------
t0 = time.time()
E0 = reduce_x(zs[0])
print("\n--- Elimination of b_{M-3}, ..., b_0 ---")
if not free_variables(E0):
    F, info = E0, "no free variables left"
else:
    F, info = eliminate_free_variables(E0)

if F is None:
    print("No equation only in u1 was found:", info)
else:
    print(f"({info}; time: {time.time() - t0:.1f} s; {len(F.monomials())} terms)")
    print(f"\nEquation in u1 only (x, y, t{M}):\n    {F} = 0")
