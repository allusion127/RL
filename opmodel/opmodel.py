"""Operating-point model: equilibrium cycle length and CBC for a fuel PAIR at a
given feed, from lattice k(BU) curves only.

Two pieces, both CPU / read-only:

CYCLEN -- linear-reactivity equilibrium.  Two variants:
  'B1'   the form validated by the bootstrap diagnosis:
         B1 = all-fresh critical burnup (core-average rho == rho*),
         Bc = 2*B1/(n+1),  n = 241/feed.
  'EOC'  solve the DISCRETE equilibrium EOC criticality directly on the true
         (curved) rho(BU): with k = floor(241/feed) full batches plus a
         remainder batch,  sum_j w_j * rho(j*Bc) / 241 == rho*.
         Same physics, no linearity assumption -- strictly the better model
         when the Gd hump makes rho(BU) non-linear over the first cycle.

CBC -- boron is the reactivity the core has to hold DOWN, so
         CBC(t) = (rho_core_op(t) - rho*) / w_B
       with rho_core_op(t) the equilibrium core-average k-inf reactivity at
       time t into the cycle, and w_B the effective differential boron worth.
       CBC_max = max over t.  w_B (and optionally rho*) are FITTED to the
       measured operating points; residuals are reported, never hidden.

The BU=0 lattice point is DROPPED everywhere: it is Xe-free, while MASTER runs
equilibrium Xe.  BU>=0.2 is the physically comparable state (and it is also the
only place the lat1600 surrogate disagrees with DeCART by more than ~100 pcm).
"""
import numpy as np

RATE = 3983.0 / 104.8 / 1000.0      # MWd/kgHM per EFPD
NSLOT = 241
W_CY1 = (129.0, 112.0)              # cy1 all-fresh full-core census (X0, X1)
RHO_STAR = 0.0168                   # calibrated critical threshold (+-0.0025)


# --------------------------------------------------------------------- curves
class Curve:
    """rho(BU) for one assembly type, Xe-free BU=0 point removed."""

    __slots__ = ("bu", "rho", "name")

    def __init__(self, bu, k, name=""):
        bu = np.asarray(bu, float)
        k = np.asarray(k, float)
        m = bu >= 0.2
        self.bu = bu[m]
        self.rho = 1.0 - 1.0 / k[m]
        self.name = name

    def __call__(self, x):
        return np.interp(x, self.bu, self.rho)


def mix(curves, weights):
    """Weighted-average rho of a fuel mixture -> callable of BU."""
    w = np.asarray(weights, float)
    w = w / w.sum()

    def f(x):
        x = np.asarray(x, float)
        out = np.zeros(x.shape if x.shape else ())
        for c, wi in zip(curves, w):
            out = out + wi * c(x)
        return out
    return f


# ------------------------------------------------------------------- batching
def batch_weights(feed):
    """Equilibrium batch census: list of (n_assemblies, batch_index).

    batch_index j means 'this batch has burnup t + j*Bc' at time t into the
    cycle (j=0 is the fresh feed).  Handles the non-integer residence of the
    1+4N feed grid exactly (241 = k*feed + remainder)."""
    out, left, j = [], NSLOT, 0
    while left > 0:
        take = min(feed, left)
        out.append((float(take), j))
        left -= take
        j += 1
    return out


def residence(feed):
    return NSLOT / float(feed)


# --------------------------------------------------------------------- cyclen
def b1_allfresh(rmix, rho_star=RHO_STAR, lo=1.0, hi=75.0):
    """All-fresh critical burnup: core-average rho crosses rho*."""
    for _ in range(90):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if float(rmix(m)) > rho_star else (lo, m)
    return 0.5 * (lo + hi)


def bc_eoc(rmix, feed, rho_star=RHO_STAR, lo=1.0, hi=60.0):
    """Discrete equilibrium cycle burnup: EOC core-average rho == rho*."""
    bw = batch_weights(feed)

    def f(bc):
        return sum(n * float(rmix((j + 1) * bc)) for n, j in bw) / NSLOT

    for _ in range(90):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) > rho_star else (lo, m)
    return 0.5 * (lo + hi)


def cyclen(rmix, feed, rho_star=RHO_STAR, variant="EOC"):
    """Equilibrium cycle length in EFPD."""
    if variant == "B1":
        b1 = b1_allfresh(rmix, rho_star)
        bc = 2.0 * b1 / (residence(feed) + 1.0)
    elif variant == "EOC":
        bc = bc_eoc(rmix, feed, rho_star)
    else:
        raise ValueError(variant)
    return bc / RATE, bc


# ------------------------------------------------------------------------ CBC
def rho_op(rmix, feed, bc, t):
    """Equilibrium core-average k-inf reactivity at burnup-time t into cycle."""
    return sum(n * float(rmix(t + j * bc)) for n, j in batch_weights(feed)) / NSLOT


def rho_op_peak(rmix, feed, bc, ngrid=41):
    """max over the cycle -- where the boron requirement peaks."""
    ts = np.linspace(0.0, bc, ngrid)
    vals = [rho_op(rmix, feed, bc, t) for t in ts]
    j = int(np.argmax(vals))
    return vals[j], float(ts[j])


def cbc_from_rho(rpk, w_b, rho_star=RHO_STAR):
    return (rpk - rho_star) / w_b


# ------------------------------------------------------------- one-shot solver
def operating_point(curves, weights, feed, rho_star=RHO_STAR, w_b=None,
                    variant="EOC"):
    """-> dict(cyclen_efpd, bc, rho_boc, rho_peak, t_peak, cbc)"""
    rm = mix(curves, weights)
    cy, bc = cyclen(rm, feed, rho_star, variant)
    rpk, tpk = rho_op_peak(rm, feed, bc)
    rboc = rho_op(rm, feed, bc, 0.0)
    d = dict(cyclen=cy, bc=bc, rho_boc=rboc, rho_peak=rpk, t_peak=tpk,
             n_res=residence(feed))
    if w_b is not None:
        d["cbc"] = cbc_from_rho(rpk, w_b, rho_star)
    return d
