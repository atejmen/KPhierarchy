r'''
It obtains the M-th equation of the truncated Lax hierarchy with a Rosenfeld-Groebner
differential elimination (own implementation on Sage polynomial rings, one variable per jet;
it does not use the elimination of dalgebra).

    L = D + u1 D^-1 + ... + uM D^-M,            D = d/dx1
    d_{x_k} L = [B_k, L],                       B_k = (L^k)_+      (k = 2 and k = M)

The exact equations of the system are built (with their non linear terms) and the
Rosenfeld-Groebner algorithm is run with the elimination ranking  uM > ... > u2 > u1.
The output is a list of regular differential chains (cases); the equations only in u1 of each
chain are printed.

Usage:    sage satoKPeliminationRG.py M [--all] [-v]

    M       integer >= 3                                           [by default 3]
    --all   also explore the cases where an initial or a separant vanishes
            (by default only the generic component is computed)
    -v      progress information on stderr

The jets are the variables of a polynomial ring; only the jets of total order <= K exist in it.
If the algorithm needs a higher one, the computation restarts with a larger K automatically.

WARNING: for M >= 5 the intermediate polynomials can be huge and the computation may take a
very long time or exhaust the memory.
'''
import os, sys, time, itertools
from math import gcd

DALGEBRA_PARENT = "/home/sage/dalgebra-develop"
sys.path.insert(0, DALGEBRA_PARENT)

from sage.all import QQ, ZZ, PolynomialRing, binomial
from dalgebra import DifferentialRing, DifferentialPolynomialRing

# ===== INPUT PARAMETERS =====
ARGS = [x for x in sys.argv[1:] if not x.startswith("-")]
ALL_CASES = "--all" in sys.argv
VERBOSE = "-v" in sys.argv
try:
    M = int(ARGS[0]) if ARGS else 3
    if M < 3:
        raise ValueError
except ValueError:
    sys.exit("Usage: sage satoKPeliminationRG.py M [--all] [-v]   (M integer >= 3)")
FLOWS = [2, M]
TIMES = [1] + FLOWS          # times involved: x1 and x_k for each flow

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
# 3. Exact Lax equations
# ----------------------------------------------------------------------------
L = PseudoDiffOp({1: DPR.one(), **{-(j + 1): u[j] for j in range(M)}})

def lax_equations(L):
    r'''
    Equations  D_k u_j = D^-j coefficient in [B_k, L].
    As L is truncated in u_M, the D^-j coefficient in [B_k, L] only is
    exact if j < M + 2 - k (otherwise, u_{M+1} would be missing). For M = 3 the resulting terms are
    D2 u1, D2 u2 and D3 u1.
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
print(f"M={M}  flows={FLOWS}  times={TIMES}  |  {len(pdes)} equations, {M} fields")
print("--- Lax equations ---")
for eq in pdes:
    print("  ", eq, "= 0")

# ----------------------------------------------------------------------------
# 4. Rosenfeld-Groebner differential elimination on Sage polynomial rings
# ----------------------------------------------------------------------------
# Every jet u_f[a] (field f, derivative orders a = (a1, ..., am)) is a variable of a Sage
# polynomial ring, so that products, gcd, exact division and factorization are done by
# Singular. The ring only contains the jets with total order <= K; if the algorithm needs a
# higher jet, JetBoundError is raised and the computation is restarted with a larger K.
NDER = len(TIMES)            # number of derivations
INITIAL_K = max(8, 2 * M)    # initial bound for the total order of the jets
MAX_FACTOR_TERMS = 3000      # polynomials with more terms are not factored


class JetBoundError(Exception):
    pass


def rank_key(jet):
    r'''Elimination ranking (higher field first), orderly inside each field.'''
    return (jet[0], sum(jet[1]), jet[1])


class JetRing:
    r'''Polynomial ring with one variable per jet (field, orders), orders of total order <= K.
    The variables are sorted by increasing ranking.'''

    def __init__(self, nfields, nder, K):
        self.K = K
        jets = [(f, a) for f in range(nfields)
                for a in itertools.product(range(K + 1), repeat=nder) if sum(a) <= K]
        jets.sort(key=rank_key)
        names = ["w%d_%s" % (f, "_".join(str(x) for x in a)) for (f, a) in jets]
        self.ring = PolynomialRing(QQ, names=names)
        self.gens = self.ring.gens()
        self.jets = jets
        self.index = {j: i for i, j in enumerate(jets)}
        self.jet_of = {g: j for g, j in zip(self.gens, jets)}
        self.rank = {g: i for i, g in enumerate(self.gens)}

    def gen(self, jet):
        i = self.index.get(jet)
        if i is None:
            raise JetBoundError(jet)
        return self.gens[i]

    def shifted(self, g, k):
        f, a = self.jet_of[g]
        b = list(a)
        b[k] += 1
        return self.gen((f, tuple(b)))


JR = None        # current JetRing (set in section 5)


# ----------------------------------------------------------------------------
# 4.1 Translation between d-polynomials of dalgebra and polynomials of the JetRing
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
    key = {(field index, order tuple): exponent}.'''
    for mon in f.monomials():
        key = {(int(gi), _orders(o)): int(e) for (gi, o), e in mon._variables.items()}
        if key:
            coeff = f.coefficient_full(mon).constant_coefficient()
        else:
            coeff = f.constant_coefficient()
        yield key, _to_QQ(coeff)


def to_ring_poly(f):
    r'''d-polynomial of dalgebra -> polynomial of the JetRing.'''
    res = JR.ring.zero()
    for key, coeff in _terms(f):
        mon = JR.ring(coeff)
        for jet, e in key.items():
            mon *= JR.gen(jet) ** e
        res += mon
    return res


def to_dpoly(p):
    r'''Polynomial of the JetRing -> d-polynomial of dalgebra.'''
    res = ZERO
    for mon, c in zip(p.monomials(), p.coefficients()):
        term = DPR(c)
        for v in mon.variables():
            gi, o = JR.jet_of[v]
            term *= GENS[gi][o] ** int(mon.degree(v))
        res += term
    return res


# ----------------------------------------------------------------------------
# 4.2 Basic operations on jet polynomials
# ----------------------------------------------------------------------------
def leader(p):
    return max(p.variables(), key=JR.rank.__getitem__)


def coeff_of(p, v, e):
    r'''Coefficient of v^e in p (a polynomial that does not contain v).'''
    return p.coefficient({v: e})


def initial(p):
    v = leader(p)
    return coeff_of(p, v, p.degree(v))


def separant(p):
    return p.derivative(leader(p))


def total_D(p, k):
    r'''Total derivative of p with respect to the k-th derivation.'''
    res = JR.ring.zero()
    for v in p.variables():
        res += p.derivative(v) * JR.shifted(v, k)
    return res


def normalize(p):
    r'''Primitive integer coefficients and positive leading coefficient (canonical form).'''
    if p == 0:
        return p
    den = 1
    for c in p.coefficients():
        d = int(c.denominator())
        den = den * d // gcd(den, d)
    p = p * den
    g = 0
    for c in p.coefficients():
        g = gcd(g, abs(int(c.numerator())))
    p = p * (QQ(1) / g)
    if p.lc() < 0:
        p = -p
    return p


def exact_div(p, f):
    r'''Exact quotient p / f, or None if f does not divide p.'''
    if not set(f.variables()) <= set(p.variables()):
        return None
    q, r = p.quo_rem(f)
    return q if r == 0 else None


def strip_known(p, S):
    r'''Removes from p the known (non vanishing) factors in S, with multiplicity.'''
    changed = True
    while changed:
        changed = False
        for f in S:
            q = exact_div(p, f)
            if q is not None:
                p, changed = q, True
    return p


def factor_poly(p):
    r'''Non constant irreducible factors of p (normalized), without multiplicities.'''
    p = normalize(p)
    if p.is_constant():
        return []
    if len(p.monomials()) > MAX_FACTOR_TERMS:
        return [p]
    try:
        return [normalize(g) for g, _ in p.factor() if not g.is_constant()]
    except Exception:                         # without factorization p itself is used
        return [p]


# ----------------------------------------------------------------------------
# 4.3 Differential reduction
# ----------------------------------------------------------------------------
def prem(p, q, v):
    r'''Pseudo-remainder of p by q with respect to the variable v.'''
    dq = q.degree(v)
    iq = coeff_of(q, v, dq)
    while True:
        dp = p.degree(v)
        if dp < dq:
            return p
        cp = coeff_of(p, v, dp)
        p = iq * p - cp * v ** (dp - dq) * q


_DERIVE_CACHE = {}


def derive(a, theta):
    r'''D^theta a (cached).'''
    key = (a, theta)
    if key in _DERIVE_CACHE:
        return _DERIVE_CACHE[key]
    q = a
    for k, n in enumerate(theta):
        for _ in range(n):
            q = total_D(q, k)
    _DERIVE_CACHE[key] = q
    return q


def deriv_order(jet, leader_jet):
    r'''theta such that jet = D^theta leader_jet, or None.'''
    if jet[0] != leader_jet[0]:
        return None
    th = tuple(x - y for x, y in zip(jet[1], leader_jet[1]))
    return None if any(t < 0 for t in th) else th


def reduce_poly(p, A):
    r'''Complete reduction of p with respect to the chain A.'''
    if not A:
        return p
    leaders = [leader(a) for a in A]
    ljets = [JR.jet_of[v] for v in leaders]
    progress = True
    while progress and p != 0:
        progress = False
        for var in sorted(p.variables(), key=JR.rank.__getitem__, reverse=True):
            vjet = JR.jet_of[var]
            for a, lj in zip(A, ljets):
                th = deriv_order(vjet, lj)
                if th is None:
                    continue
                if sum(th) == 0:
                    if p.degree(var) < a.degree(var):
                        continue
                    q = a
                else:
                    q = derive(a, th)
                p = prem(p, q, var)
                progress = True
                break
            if progress:
                break
    return p


def delta_poly(a, b):
    r'''Delta-polynomial of the pair (a, b), or None if their leaders do not interact.'''
    ja, jb = JR.jet_of[leader(a)], JR.jet_of[leader(b)]
    if ja[0] != jb[0]:
        return None
    gam = tuple(max(x, y) for x, y in zip(ja[1], jb[1]))
    tha = tuple(g - x for g, x in zip(gam, ja[1]))
    thb = tuple(g - y for g, y in zip(gam, jb[1]))
    if sum(tha) == 0 or sum(thb) == 0:
        return None
    return separant(b) * derive(a, tha) - separant(a) * derive(b, thb)


# ----------------------------------------------------------------------------
# 4.4 Rosenfeld-Groebner with case splitting
# ----------------------------------------------------------------------------
class Branch:
    r'''A = chain (autoreduced), D = pending polynomials, S = factors assumed non vanishing,
    nosep = polynomials whose initial/separant splitting was already done.'''

    def __init__(self, A, D, S, nosep):
        self.A, self.D, self.S, self.nosep = A, D, S, nosep

    def copy(self):
        return Branch(list(self.A), list(self.D), set(self.S), set(self.nosep))


def _new_factors(p, S):
    return [f for f in factor_poly(p) if f not in S]


def rosenfeld_groebner(F, verbose=False, all_cases=False, on_result=None):
    r'''
    Rosenfeld-Groebner decomposition of the system F = 0 (list of polynomials of the JetRing)
    with the elimination ranking. Returns the list of branches (A = coherent autoreduced
    chain, S = inequations). all_cases=False: the branches where an initial or a separant
    vanishes are discarded (generic component).
    '''
    stack = [Branch([], [normalize(f) for f in F if f != 0], set(), set())]
    results, steps = [], 0
    while stack:
        br = stack.pop()
        while br.D:
            steps += 1
            p = reduce_poly(br.D.pop(), br.A)
            if p == 0:
                continue
            p = strip_known(normalize(p), br.S)
            if p.is_constant():                           # inconsistent branch
                br = None
                break
            facs = [f for f in factor_poly(p) if f not in br.S]
            if not facs:                                  # p is a product of non vanishing factors
                br = None
                break
            if len(facs) > 1:                             # split into disjoint cases f_k = 0
                for idx, f in enumerate(facs):
                    nb = br.copy()
                    nb.D.append(f)
                    nb.S.update(facs[:idx])
                    stack.append(nb)
                br = None
                break
            p = facs[0]
            lead = leader(p)
            d = p.degree(lead)
            ini, sep = initial(p), separant(p)
            ini_f = _new_factors(ini, br.S)
            if all_cases and ini_f and p not in br.nosep:             # case initial = 0
                nb = br.copy()
                tail = p - ini * lead ** d
                nb.D.append(normalize(ini))
                if tail != 0:
                    nb.D.append(tail)
                nb.nosep.add(p)
                stack.append(nb)
            if all_cases and d > 1 and p not in br.nosep:             # case separant = 0
                if _new_factors(sep, br.S):
                    nb = br.copy()
                    nb.D.append(normalize(sep))
                    nb.D.append(p)
                    nb.nosep.add(p)
                    stack.append(nb)
            br.S.update(ini_f + _new_factors(sep, br.S))              # generic case: ini, sep != 0
            # insert p in the chain; the elements reducible by p go back to D
            ljet = JR.jet_of[lead]
            kept, moved = [], []
            for a in br.A:
                th = deriv_order(JR.jet_of[leader(a)], ljet)
                if th is not None and (sum(th) > 0 or a.degree(lead) >= d):
                    moved.append(a)
                else:
                    kept.append(a)
            for a in kept:
                dl = delta_poly(a, p)
                if dl is not None and dl != 0:
                    br.D.append(dl)
            br.D.extend(moved)
            br.A = kept + [p]
        if br is not None:
            results.append(br)
            if on_result is not None:
                on_result(len(results) - 1, br)
        if verbose:
            print(f"  [rg] stack={len(stack)} chains={len(results)} steps={steps}", file=sys.stderr)
    return results


# ----------------------------------------------------------------------------
# 5. Run: elimination of u2, ..., uM
# ----------------------------------------------------------------------------
print(f"\n--- Rosenfeld-Groebner (ranking u{M} > ... > u2 > u1, "
      f"{'all cases' if ALL_CASES else 'generic component'}) ---")
t0 = time.time()
u1_equations = []


def report(i, chain):
    only_u1 = [a for a in chain.A if all(JR.jet_of[v][0] == 0 for v in a.variables())]
    print(f"\nchain {i}: {len(chain.A)} equations, {len(only_u1)} only in u1 "
          f"(t = {time.time() - t0:.1f} s)", flush=True)
    for a in only_u1:
        eq = to_dpoly(a)
        u1_equations.append(eq)
        print("   ", eq, "= 0", flush=True)


K = INITIAL_K
while True:
    JR = JetRing(M, NDER, K)
    _DERIVE_CACHE.clear()
    u1_equations = []
    print(f"(jet ring: {len(JR.jets)} variables, total order <= {K})", flush=True)
    try:
        chains = rosenfeld_groebner([to_ring_poly(eq) for eq in pdes], verbose=VERBOSE,
                                    all_cases=ALL_CASES, on_result=report)
        break
    except JetBoundError as exc:
        K += 4
        print(f"Jet {exc} is out of the ring: restarting with total order <= {K}", flush=True)

print(f"\n(time: {time.time() - t0:.1f} s, {len(chains)} chain(s))")

if not u1_equations:
    print("No equation only in u1 was found.")
else:
    print(f"{len(u1_equations)} equation(s) in u1 were found.")
