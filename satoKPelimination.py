r'''
It obtains the M equation of Lax truncated hierarchy with differential elimination of dalgebra.

    L = D + u1 D^-1 + ... + uM D^-M,            D = d/dx1
    d_{x_k} L = [B_k, L],                       B_k = (L^k)_+      (k = 2 y k = M)

The exact equations of the system are built (con their no linear terms) and 
DSystem.eliminate_variables(u2, ..., uM) is called to obtain equations just in u1.

Usage:    sage lax_eliminacion.py M [bound_L] [alg_res]

    M       integer >= 3                                           [by default 3]
    bound_L  maximum order of prolongation of the equations        [by default 4]
    alg_res  "auto" | "algebraic" | "autoreduced" | "iterative"    [by default "algebraic"]

WARNING: The elimination may last a long time or find out nothing if bound_L is small;
in that case, a message is printed.
'''
import os, sys, time

DALGEBRA_PARENT = "/home/sage/dalgebra-develop"
sys.path.insert(0, DALGEBRA_PARENT)

from sage.all import QQ, ZZ, PolynomialRing, binomial
from dalgebra import DifferentialRing, DifferentialPolynomialRing
try:
    from dalgebra import DifferentialSystem
except ImportError:
    from dalgebra.dpolynomial.dsystems import DifferentialSystem

# ===== INPUT PARAMETERS =====
try:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    BOUND_L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ALG_RES = sys.argv[3] if len(sys.argv) > 3 else "algebraic"
    if M < 3 or BOUND_L < 1 or ALG_RES not in ("auto", "algebraic", "autoreduced", "iterative"):
        raise ValueError
except ValueError:
    sys.exit("Usage: sage lax_eliminacion.py M [bound_L] [alg_res]   (M >= 3, bound_L >= 1, "
             "alg_res in auto|algebraic|autoreduced|iterative)")
FLOWS = [2, M]
TIMES = [1] + FLOWS          # times involved: x1 y x_k for each flow

# ----------------------------------------------------------------------------
# 1. Differential Ring with TIMES derivatives and d-polynomials Ring
# ----------------------------------------------------------------------------
P = PolynomialRing(QQ, names=[f'x{t}' for t in TIMES])
R = DifferentialRing(P, *P.derivation_module().gens())   # d/dx1, d/dx_k, ...
DPR = DifferentialPolynomialRing(R, *[f'u{j}' for j in range(1, M + 1)])

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
# 3. Ecuaciones de Lax exactas
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
print(f"M={M}  flujos={FLOWS}  tiempos={TIMES}  |  {len(pdes)} equations, {M} fields")
print("--- Lax equations ---")
for eq in pdes:
    print("  ", eq, "= 0")

# ----------------------------------------------------------------------------
# 4. Differential elimination of dalgebra: eliminating u2, ..., uM
# ----------------------------------------------------------------------------
dmsystem = DifferentialSystem(pdes, DPR)
to_eliminate = [f'u{j}' for j in range(2, M + 1)]

print(f"\n--- Eliminating {to_eliminate} (bound_L={BOUND_L}, alg_res='{ALG_RES}') ---")
t0 = time.time()
try:
    result = dmsystem.eliminate_variables(*to_eliminate, bound_L=BOUND_L, alg_res=ALG_RES)
except NotImplementedError as exc:
    sys.exit(f"Elimination method not available in this dalgebra version: {exc}")
print(f"(tiempo: {time.time() - t0:.1f} s)")

u1_equations = list(result.equations()) if result.size() > 0 else []

if not u1_equations:
    print("None equation just in u1 founded.")
    print("Try with a greater bound_L, o with other alg_res.")
    sys.exit(0)

print(f"{len(u1_equations)} equation(s) in u1 were founded:")
for i, ec in enumerate(u1_equations):
    print(f"\n[{i}]  {ec} = 0")