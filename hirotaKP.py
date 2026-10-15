r'''
M-th equation of the KP hierarchy in Hirota form and in terms of u1.

Step 1 (tau). Sato's formula in Hirota derivatives D_i (i = 1, ..., M):

    sum_{k+n=p} (-2)^k / (k! n!) * p_{k+1}(D~) * D_1^n (tau . tau) = 0,   p = M,

where D~ = (D_1, D_2/2, D_3/3, ...) and p_m are the elementary Schur polynomials,
exp(sum_i t_i z^i) = sum_m p_m(t) z^m. (The odd terms vanish on tau . tau.)
For M = 3 it is the KP equation and for M = 4 the next one of the hierarchy.

Step 2 (u1). With tau = exp(phi) the equation becomes an equation in phi, and
u1 = d^2 phi / dx1^2. The equation in u1 only is obtained by differential elimination
(Rosenfeld-Groebner with the ranking phi > u1) of the system

    { equation in phi = 0,  u1 - phi_{x1 x1} = 0 }.

Usage:    sage hirotaKP.py M [--all] [-v]

    M       integer >= 3                                           [by default 3]
    --all   also explore the cases where an initial or a separant vanishes
            (by default only the generic component is computed)
    -v      progress information on stderr
'''
import sys
import time
import itertools
from fractions import Fraction
from math import gcd, factorial

# ===== INPUT PARAMETERS =====
ARGS = [x for x in sys.argv[1:] if not x.startswith("-")]
ALL_CASES = "--all" in sys.argv
VERBOSE = "-v" in sys.argv
try:
    M = int(ARGS[0]) if ARGS else 3
    if M < 3:
        raise ValueError
except ValueError:
    sys.exit("Usage: sage hirotaKP.py M [--all] [-v]   (M integer >= 3)")

NDER = M                          # derivations: x1, ..., xM
TIMES = list(range(1, M + 1))
NAMES = ["tau"]                   # names of the fields, used only to print
MAX_FACTOR_VARS = 10              # polynomials with more variables are not factored
ZERO_ORD = (0,) * NDER

# ----------------------------------------------------------------------------
# 1. Polynomials in jets
# ----------------------------------------------------------------------------
# A jet is a pair (field, orders), orders = tuple of derivative orders with respect to
# x1, ..., xM.  A monomial is a tuple of (jet, exponent) sorted by ranking.
def rank_key(v):
    r'''Elimination ranking (higher field first), orderly inside each field.'''
    return (v[0], sum(v[1]), v[1])

def shift(v, k):
    a = list(v[1])
    a[k] += 1
    return (v[0], tuple(a))

def _mulmon(m1, m2):
    if not m1:
        return m2
    if not m2:
        return m1
    d = dict(m1)
    for v, e in m2:
        d[v] = d.get(v, 0) + e
    return tuple(sorted(d.items(), key=lambda t: rank_key(t[0])))

def _mon_key(m):
    return tuple((rank_key(w), e) for w, e in reversed(m))

class JetPoly:
    r'''Sparse polynomial in jets with rational coefficients.'''
    __slots__ = ('d', 'cache')

    def __init__(self, d=None):
        self.d = d if d is not None else {}
        self.cache = {}

    @staticmethod
    def const(c):
        c = Fraction(c)
        return JetPoly({(): c}) if c != 0 else JetPoly()

    @staticmethod
    def var(v):
        return JetPoly({((v, 1),): Fraction(1)})

    def is_zero(self):
        return not self.d

    def is_const(self):
        return all(m == () for m in self.d)

    def __add__(self, o):
        if not isinstance(o, JetPoly):
            o = JetPoly.const(o)
        r = dict(self.d)
        for m, c in o.d.items():
            nc = r.get(m, 0) + c
            if nc:
                r[m] = nc
            else:
                r.pop(m, None)
        return JetPoly(r)

    __radd__ = __add__

    def __neg__(self):
        return JetPoly({m: -c for m, c in self.d.items()})

    def __sub__(self, o):
        if not isinstance(o, JetPoly):
            o = JetPoly.const(o)
        return self + (-o)

    def __mul__(self, o):
        if not isinstance(o, JetPoly):
            o = Fraction(o)
            return JetPoly({m: c * o for m, c in self.d.items()}) if o != 0 else JetPoly()
        r = {}
        for m1, c1 in self.d.items():
            for m2, c2 in o.d.items():
                m = _mulmon(m1, m2)
                v = r.get(m, 0) + c1 * c2
                if v:
                    r[m] = v
                else:
                    r.pop(m, None)
        return JetPoly(r)

    __rmul__ = __mul__

    def __pow__(self, n):
        res = JetPoly.const(1)
        for _ in range(n):
            res = res * self
        return res

    def vars(self):
        return {w for m in self.d for (w, _) in m}

    def leader(self):
        return max(self.vars(), key=rank_key)

    def deg(self, v):
        return max((e for m in self.d for (w, e) in m if w == v), default=0)

    def coeffs(self, v):
        r'''Coefficients of the powers of v: {exponent: JetPoly without v}.'''
        res = {}
        for m, c in self.d.items():
            e, rest = 0, []
            for (w, ee) in m:
                if w == v:
                    e = ee
                else:
                    rest.append((w, ee))
            res.setdefault(e, {})[tuple(rest)] = c
        return {e: JetPoly(dd) for e, dd in res.items()}

    def initial(self):
        v = self.leader()
        return self.coeffs(v)[self.deg(v)]

    def diff(self, v):
        r'''Partial derivative with respect to the jet v.'''
        r = {}
        for m, c in self.d.items():
            for idx, (w, e) in enumerate(m):
                if w == v:
                    nm = m[:idx] + (((w, e - 1),) if e > 1 else ()) + m[idx + 1:]
                    r[nm] = r.get(nm, 0) + c * e
        return JetPoly({m: c for m, c in r.items() if c})

    def separant(self):
        return self.diff(self.leader())

    def D(self, k):
        r'''Total derivative with respect to the k-th derivation.'''
        r = {}
        for m, c in self.d.items():
            for idx, (w, e) in enumerate(m):
                base = m[:idx] + (((w, e - 1),) if e > 1 else ()) + m[idx + 1:]
                nm = _mulmon(base, ((shift(w, k), 1),))
                r[nm] = r.get(nm, 0) + c * e
        return JetPoly({m: c for m, c in r.items() if c})

    def normalized(self):
        r'''Primitive integer coefficients and canonical sign (to compare and to factor).'''
        if not self.d:
            return self
        den = 1
        for c in self.d.values():
            den = den * c.denominator // gcd(den, c.denominator)
        ints = {m: int(c * den) for m, c in self.d.items()}
        g = 0
        for v in ints.values():
            g = gcd(g, abs(v))
        top = max(ints, key=_mon_key)
        sgn = 1 if ints[top] > 0 else -1
        return JetPoly({m: Fraction(sgn * v // g) for m, v in ints.items()})

    def key(self):
        return tuple(sorted(self.normalized().d.items(), key=lambda t: t[0]))

def var_str(v):
    f, a = v
    sub = " ".join((f"x{t}" if n == 1 else f"x{t}^{n}") for t, n in zip(TIMES, a) if n)
    return NAMES[f] + (f"_{{{sub}}}" if sub else "")

def poly_str(p):
    if p.is_zero():
        return "0"
    items = sorted(p.d.items(), key=lambda t: _mon_key(t[0]), reverse=True)
    out = []
    for m, c in items:
        facs = "*".join(var_str(w) + (f"^{e}" if e > 1 else "") for w, e in m)
        cs = str(abs(c)) if abs(c).denominator == 1 else f"({abs(c)})"
        term = facs if (facs and abs(c) == 1) else (cs + ("*" + facs if facs else ""))
        out.append(("- " if c < 0 else "+ ") + term)
    s = " ".join(out)
    return s[2:] if s.startswith("+ ") else "-" + s[2:].lstrip()

# ----------------------------------------------------------------------------
# 2. Step 1: the equation in tau (Sato's formula)
# ----------------------------------------------------------------------------
# Hirota derivatives D_i are represented as the jets (0, e_i) of a polynomial ring.
def hirota_var(i):
    return JetPoly.var((0, tuple(1 if j == i - 1 else 0 for j in range(NDER))))

def schur_polynomials(mmax):
    r'''p_m(D~), m = 0, ..., mmax, with D~ = (D_1, D_2/2, ...); m p_m = sum_i D_i p_{m-i}.
    D_i with i > M is omitted: it only appears alone (an odd term, zero on tau . tau).'''
    p = [JetPoly.const(1)]
    for m in range(1, mmax + 1):
        acc = JetPoly()
        for i in range(1, min(m, M) + 1):
            acc = acc + hirota_var(i) * p[m - i]
        p.append(acc * Fraction(1, m))
    return p

def hirota_operator(order):
    r'''sum_{k+n=order} (-2)^k / (k! n!) p_{k+1}(D~) D_1^n, keeping the even monomials.'''
    p = schur_polynomials(order + 1)
    total = JetPoly()
    for k in range(order + 1):
        n = order - k
        total = total + p[k + 1] * hirota_var(1) ** n * Fraction((-2) ** k, factorial(k) * factorial(n))
    even = {m: c for m, c in total.d.items() if sum(e for _, e in m) % 2 == 0}
    return JetPoly(even)

def _exponents(mon):
    a = [0] * NDER
    for (_, o), e in mon:
        a[o.index(1)] += e
    return tuple(a)

def operator_str(op):
    r'''The operator polynomial with integer coefficients, e.g. D1^4 - 4*D1*D3 + 3*D2^2.'''
    op = op.normalized()
    terms = []
    for mon, c in op.d.items():
        a = _exponents(mon)
        terms.append((a, c))
    terms.sort(key=lambda t: (-sum(t[0]), tuple(-x for x in t[0])))
    out = []
    for a, c in terms:
        facs = "*".join(f"D{i + 1}" + (f"^{e}" if e > 1 else "") for i, e in enumerate(a) if e)
        cs = "" if abs(c) == 1 else f"{abs(c)}*"
        out.append(("- " if c < 0 else "+ ") + cs + facs)
    s = " ".join(out)
    return s[2:] if s.startswith("+ ") else "-" + s[2:].lstrip()

def binom(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def tau_jet(beta):
    return JetPoly.var((0, tuple(beta)))

def apply_hirota(alpha):
    r'''D^alpha (tau . tau) = sum_{beta <= alpha} (-1)^{|alpha - beta|} prod binom(alpha_i, beta_i)
    tau_beta tau_{alpha - beta}   (polynomial in the jets of tau).'''
    res = JetPoly()
    for beta in itertools.product(*[range(a + 1) for a in alpha]):
        sign = (-1) ** (sum(alpha) - sum(beta))
        coef = 1
        for a, b in zip(alpha, beta):
            coef *= binom(a, b)
        gamma = tuple(a - b for a, b in zip(alpha, beta))
        res = res + tau_jet(beta) * tau_jet(gamma) * (sign * coef)
    return res

def tau_equation(op):
    res = JetPoly()
    for mon, c in op.d.items():
        res = res + apply_hirota(_exponents(mon)) * c
    return res

# ----------------------------------------------------------------------------
# 3. Step 2a: tau = exp(phi); the equation in phi
# ----------------------------------------------------------------------------
# tau_beta / tau = Y_beta(phi) with Y_0 = 1 and Y_{beta + e_i} = D_i Y_beta + phi_{e_i} Y_beta.
# phi is the field 1 (u1 is the field 0).
_Y = {}

def Y(beta):
    if beta in _Y:
        return _Y[beta]
    if sum(beta) == 0:
        res = JetPoly.const(1)
    else:
        i = next(j for j, b in enumerate(beta) if b > 0)
        prev = list(beta)
        prev[i] -= 1
        yp = Y(tuple(prev))
        e_i = tuple(1 if j == i else 0 for j in range(NDER))
        res = yp.D(i) + JetPoly.var((1, e_i)) * yp
    _Y[beta] = res
    return res

def phi_equation(teq):
    r'''Substitutes tau_beta tau_gamma -> Y_beta Y_gamma (division by tau^2).'''
    res = JetPoly()
    for mon, c in teq.d.items():
        term = JetPoly.const(c)
        for (_, beta), e in mon:
            term = term * Y(beta) ** e
        res = res + term
    return res

# ----------------------------------------------------------------------------
# 4. Step 2b: Rosenfeld-Groebner differential elimination (phi > u1)
# ----------------------------------------------------------------------------
def exact_div(p, f):
    r'''Exact quotient p / f (lexicographic order induced by the ranking), or None.'''
    if f.is_const():
        c = f.d[()]
        return JetPoly({m: v / c for m, v in p.d.items()})
    if not f.vars() <= p.vars():
        return None
    lf = max(f.d, key=_mon_key)
    cf = f.d[lf]
    df = dict(lf)
    q, r = {}, dict(p.d)
    while r:
        lr = max(r, key=_mon_key)
        dr = dict(lr)
        if any(dr.get(w, 0) < e for w, e in lf):
            return None
        qm = tuple(sorted(((w, dr.get(w, 0) - df.get(w, 0)) for w in set(dr) | set(df)
                           if dr.get(w, 0) - df.get(w, 0) > 0), key=lambda t: rank_key(t[0])))
        qc = r[lr] / cf
        q[qm] = qc
        for m, c in f.d.items():
            mm = _mulmon(qm, m)
            nc = r.get(mm, 0) - qc * c
            if nc:
                r[mm] = nc
            else:
                r.pop(mm, None)
    return JetPoly(q)

def strip_known(p, S):
    r'''Removes from p the known (non vanishing) factors in S, with multiplicity.'''
    changed = True
    while changed:
        changed = False
        for f in S.values():
            q = exact_div(p, f)
            if q is not None:
                p, changed = q, True
    return p

def factor_poly(p):
    r'''Non constant irreducible factors of p (normalized), without multiplicities.
    Needs sympy; without it (or for big polynomials) p itself is returned.'''
    p = p.normalized()
    if p.is_const():
        return []
    if len(p.vars()) > MAX_FACTOR_VARS:
        return [p]
    try:
        from sympy import QQ
        from sympy.polys.rings import ring
        vs = sorted(p.vars(), key=rank_key)
        idx = {v: i for i, v in enumerate(vs)}
        Rs, *_ = ring([f"v{i}" for i in range(len(vs))], QQ)
        dd = {}
        for m, c in p.d.items():
            ex = [0] * len(vs)
            for w, e in m:
                ex[idx[w]] = e
            dd[tuple(ex)] = QQ(c.numerator, c.denominator)
        _, facs = Rs.from_dict(dd).factor_list()
        res = []
        for g, _m in facs:
            if g.is_ground:
                continue
            gd = {}
            for ex, cc in g.terms():
                mon = tuple(sorted(((vs[i], e) for i, e in enumerate(ex) if e),
                                   key=lambda t: rank_key(t[0])))
                gd[mon] = Fraction(int(cc.numerator), int(cc.denominator))
            res.append(JetPoly(gd).normalized())
        return res
    except Exception:
        return [p]

def prem(p, q, v):
    r'''Pseudo-remainder of p by q with respect to the variable v.'''
    dq = q.deg(v)
    iq = q.coeffs(v)[dq]
    while True:
        dp = p.deg(v)
        if dp < dq:
            return p
        cp = p.coeffs(v)[dp]
        vpow = JetPoly({(((v, dp - dq),) if dp > dq else ()): Fraction(1)})
        p = iq * p - cp * vpow * q

def derive(a, theta):
    r'''D^theta a (cached).'''
    if theta in a.cache:
        return a.cache[theta]
    q = a
    for k, n in enumerate(theta):
        for _ in range(n):
            q = q.D(k)
    a.cache[theta] = q
    return q

def deriv_order(var, leader):
    r'''theta such that var = D^theta leader, or None.'''
    if var[0] != leader[0]:
        return None
    th = tuple(x - y for x, y in zip(var[1], leader[1]))
    return None if any(t < 0 for t in th) else th

def reduce_poly(p, A):
    r'''Complete reduction of p with respect to the chain A.'''
    if not A:
        return p
    leaders = [a.leader() for a in A]
    progress = True
    while progress and not p.is_zero():
        progress = False
        for var in sorted(p.vars(), key=rank_key, reverse=True):
            for a, la in zip(A, leaders):
                th = deriv_order(var, la)
                if th is None:
                    continue
                if sum(th) == 0:
                    if p.deg(var) < a.deg(var):
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
    la, lb = a.leader(), b.leader()
    if la[0] != lb[0]:
        return None
    gam = tuple(max(x, y) for x, y in zip(la[1], lb[1]))
    tha = tuple(g - x for g, x in zip(gam, la[1]))
    thb = tuple(g - y for g, y in zip(gam, lb[1]))
    if sum(tha) == 0 or sum(thb) == 0:
        return None
    return b.separant() * derive(a, tha) - a.separant() * derive(b, thb)

class Branch:
    r'''A = chain (autoreduced), D = pending polynomials, S = factors assumed non vanishing,
    nosep = polynomials whose initial/separant splitting was already done.'''
    def __init__(self, A, D, S, nosep):
        self.A, self.D, self.S, self.nosep = A, D, S, nosep

    def copy(self):
        return Branch(list(self.A), list(self.D), dict(self.S), set(self.nosep))

def _new_factors(p, S):
    return [f for f in factor_poly(p) if f.key() not in S]

def rosenfeld_groebner(F, verbose=False, all_cases=False, on_result=None):
    r'''
    Rosenfeld-Groebner decomposition of the system F = 0 (list of JetPoly) with the
    elimination ranking. Returns the list of branches (A = coherent autoreduced chain,
    S = inequations). all_cases=False: the branches where an initial or a separant vanishes
    are discarded (generic component).
    '''
    stack = [Branch([], [f.normalized() for f in F if not f.is_zero()], {}, set())]
    results, steps = [], 0
    while stack:
        br = stack.pop()
        while br.D:
            steps += 1
            p = reduce_poly(br.D.pop(), br.A)
            if p.is_zero():
                continue
            p = strip_known(p.normalized(), br.S)
            if p.is_const():                              # inconsistent branch
                br = None
                break
            facs = [f for f in factor_poly(p) if f.key() not in br.S]
            if not facs:                                  # p is a product of non vanishing factors
                br = None
                break
            if len(facs) > 1:                             # split into disjoint cases f_k = 0
                for idx, f in enumerate(facs):
                    nb = br.copy()
                    nb.D.append(f)
                    nb.S.update({g.key(): g for g in facs[:idx]})
                    stack.append(nb)
                br = None
                break
            p = facs[0]
            lead = p.leader()
            d = p.deg(lead)
            ini, sep = p.initial(), p.separant()
            ini_f = _new_factors(ini, br.S)
            if all_cases and ini_f and p.key() not in br.nosep:       # case initial = 0
                nb = br.copy()
                tail = p - ini * JetPoly({((lead, d),): Fraction(1)})
                nb.D.append(ini.normalized())
                if not tail.is_zero():
                    nb.D.append(tail)
                nb.nosep.add(p.key())
                stack.append(nb)
            if all_cases and d > 1 and p.key() not in br.nosep:       # case separant = 0
                if _new_factors(sep, br.S):
                    nb = br.copy()
                    nb.D.append(sep.normalized())
                    nb.D.append(p)
                    nb.nosep.add(p.key())
                    stack.append(nb)
            for f in ini_f + _new_factors(sep, br.S):                 # generic case: ini, sep != 0
                br.S[f.key()] = f
            # insert p in the chain; the elements reducible by p go back to D
            kept, moved = [], []
            for a in br.A:
                th = deriv_order(a.leader(), lead)
                if th is not None and (sum(th) > 0 or a.deg(lead) >= d):
                    moved.append(a)
                else:
                    kept.append(a)
            for a in kept:
                dl = delta_poly(a, p)
                if dl is not None and not dl.is_zero():
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
# 5. Run
# ----------------------------------------------------------------------------
t0 = time.time()
print(f"M = {M}: equation of the KP hierarchy with times x1, ..., x{M}")

# Step 1: equation in tau
op = hirota_operator(M)
print("\n--- Step 1: equation in tau (Sato's formula) ---")
print(f"Hirota form:  ( {operator_str(op)} ) tau.tau = 0")
teq = tau_equation(op)
NAMES = ["tau"]
print("In the derivatives of tau:")
print("   ", poly_str(teq.normalized()), "= 0")

# Step 2: equation in phi = log(tau) and in u1 = phi_{x1 x1}
heq = phi_equation(teq).normalized()
NAMES = ["u1", "phi"]
print("\n--- Step 2: tau = exp(phi); equation in phi ---")
print("   ", poly_str(heq), "= 0")

e1 = tuple(2 if j == 0 else 0 for j in range(NDER))
link = JetPoly.var((0, ZERO_ORD)) - JetPoly.var((1, e1))              # u1 - phi_{x1 x1}
print("\n--- Step 3: elimination of phi with u1 = phi_{x1 x1} "
      f"(Rosenfeld-Groebner, {'all cases' if ALL_CASES else 'generic component'}) ---")

u1_equations = []

def report(i, chain):
    only_u1 = [a for a in chain.A if all(v[0] == 0 for v in a.vars())]
    print(f"\nchain {i}: {len(chain.A)} equations, {len(only_u1)} only in u1 "
          f"(t = {time.time() - t0:.1f} s)", flush=True)
    for a in only_u1:
        u1_equations.append(a)
        print("   ", poly_str(a), "= 0", flush=True)

chains = rosenfeld_groebner([heq, link], verbose=VERBOSE, all_cases=ALL_CASES, on_result=report)
print(f"\n(time: {time.time() - t0:.1f} s, {len(chains)} chain(s))")

if not u1_equations:
    print("No equation only in u1 was found.")
else:
    print(f"{len(u1_equations)} equation(s) in u1 were found.")