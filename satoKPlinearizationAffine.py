r'''
KP equation of flow M

    L = D + u1 D^-1 + u2 D^-2 + uM D^-M,        D = d/dx1
    d_{x_k} L = [B_k, L],                       B_k = (L^k)_+      (k = 2, M)

Run with SageMath:

    sage satoKPlinearizationAffine.py M

Steps: Lax equations -> linear triangularization with the independent terms carried along
-> (section 7) non linear step that eliminates u2, ..., uM when possible.
'''
import os, sys

DALGEBRA_PARENT = "/home/sage/dalgebra-develop"
sys.path.insert(0, DALGEBRA_PARENT)

from sage.all import QQ, ZZ, PolynomialRing, Matrix, binomial
from dalgebra import DifferentialRing, DifferentialPolynomialRing

# ==== INPUT PARAMETER: M (KP equation number) ====
# Usage: sage satoKPlinearizationAffine.py M (M>=3; by default 3)

try:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if M < 3:
        raise ValueError
except ValueError:
    sys.exit("Usage: sage satoKPlinearizationAffine.py M (M integer >= 3)")
FLOWS = [2, M]  # flows x2 (B2), xM (BM)
TIMES = [1] + FLOWS

# ----------------------------------------------------------------------------
# 1. Differential Ring with TIMES derivatives and d-polynomials Ring
# ----------------------------------------------------------------------------
P = PolynomialRing(QQ, names=[f'x{t}' for t in TIMES])
R = DifferentialRing(P, *P.derivation_module().gens())   # d/dx1, ..., d/dxM
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

# Rows that involves only u1: T[i,0]·u1 = b[i]
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

# ----------------------------------------------------------------------------
# 7. Non linear step: elimination of u2, ..., uM
# ----------------------------------------------------------------------------
# The equations of flow x2 are   D2 u_j = u_j'' + 2 u_{j+1}' + (terms in u1, ..., u_j),
# so   u_{j+1}' = W_{j+1}(u1, ..., u_j and their derivatives)   (local, no integrals).
# Therefore every variable u_j[a,b,...] with j >= 2 and a >= 1 (at least one x1-derivative)
# can be replaced by a polynomial in lower fields.  Only the u_j with no x1-derivative
# (a = 0) cannot be replaced; call them the "free variables" z_1, ..., z_n.
# Each D1 z_i is again a local expression (by the same rule), so the set Z of free variables
# is closed under D1.  When E and every D1 z_i are AFFINE in Z, the equations
#       E_k = D1^k E = c_{k,1} z_1 + ... + c_{k,n} z_n + N_k = 0      (k = 0, 1, ..., n)
# are n+1 linear equations in n unknowns with coefficients in u1 only, and the determinant
#       F = det [ c_{k,1} ... c_{k,n} | N_k ]    (an (n+1) x (n+1) matrix)
# vanishes: F = 0 is an equation in u1 alone (no integrals).  For n = 1 this is the
# elimination D1(c)*E - c*D1(E).
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

def affine_split(f, Zkeys):
    r'''f = sum_i c_i * z_i + N with c_i, N free of the variables in Zkeys.
    Returns ([c_1, ..., c_n], N), or None if f is not affine in those variables.'''
    index = {k: i for i, k in enumerate(Zkeys)}
    cs, Nrest = [ZERO for _ in Zkeys], ZERO
    for key, coeff in _terms(f):
        mon, inside = DPR(coeff), []
        for (gi, o), e in key.items():
            if (gi, o) in index:
                inside.append(((gi, o), e))
            else:
                mon *= GENS[gi][o] ** e
        if not inside:
            Nrest += mon
        elif len(inside) == 1 and inside[0][1] == 1:
            cs[index[inside[0][0]]] += mon
        else:
            return None
    return cs, Nrest

def d1_of_free(zkey):
    r'''D1 z for z = u_{g+1}[0, b, ...]  (reduced d-polynomial).'''
    gz, oz = zkey
    return reduce_x(_Dn(W_of(gz), (0,) + oz[1:]))

def free_closure(free):
    r'''Free variables closed under D1.'''
    Z, frontier = set(free), list(free)
    while frontier:
        new = free_variables(d1_of_free(frontier.pop())) - Z
        Z |= new
        frontier.extend(new)
    return sorted(Z)

def det_commutative(Mx):
    r'''Division-free determinant (Laplace expansion with memoization over column subsets).'''
    n = len(Mx)
    memo = {}
    def minor(row, cols):
        if row == n:
            return DPR.one()
        if (row, cols) not in memo:
            total, sign = ZERO, 1
            for pos, c in enumerate(cols):
                if Mx[row][c] != 0:
                    total += sign * Mx[row][c] * minor(row + 1, cols[:pos] + cols[pos + 1:])
                sign = -sign
            memo[(row, cols)] = total
        return memo[(row, cols)]
    return minor(0, tuple(range(n)))

def eliminate_affine(E):
    r'''Equation in u1 only from E (affine in the free variables), or (None, reason).'''
    Z = free_closure(free_variables(E))
    nz = len(Z)
    Ek = [E]
    for _ in range(nz + 2):              # extra derivatives in case the first window is degenerate
        Ek.append(reduce_x(Ek[-1].derivative(0)))
    rows = []
    for f in Ek:
        split = affine_split(f, Z)
        if split is None:
            return None, "the equation is not affine in the free variables"
        rows.append(split[0] + [split[1]])
    for start in range(3):
        F = det_commutative(rows[start:start + nz + 1])
        if F != 0:
            return F, f"{nz} free variable(s): " + ", ".join(_name(*z) for z in Z)
    return None, "all the determinants vanish (degenerate system)"

def _name(gi, o):
    return f"u{gi + 1}[{','.join(map(str, o))}]"

print("\n--- Elimination of u2, ..., uM (non linear step) ---")
final_equations = []
for eq in equations_u1:
    E = reduce_x(eq)
    if not free_variables(E):
        print("Local equation in u1 (no integrals):")
        F = E
    else:
        F, info = eliminate_affine(E)
        if F is None:
            print(f"Not closed: {info}.")
            print("   ", E, "= 0")
            continue
        print(f"Eliminated {info}; local equation in u1 (no integrals):")
    final_equations.append(F)
    print("   ", F, "= 0")

# NOTE: the result is a consequence of the Lax equations, not necessarily the lowest-order one
# (it can have a large order in x1 and many terms; M = 5 gives order 11).  If the equation
# is not affine in the free variables the script reports it and prints what remains.