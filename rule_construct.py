"""RULE-ONLY LP CONSTRUCTOR -- the Phase-Construct acid test builder.

Builds complete, legal, novel feed-121 loading patterns for cell
``ga80 | E1_E2 | feed 121`` from the VALIDATED MINED RULES ALONE
(scratchpad ``rules/validated.json``, 2026-08-11: 18 CONFIRMED + 6
DOMAIN-LIMITED, E1_E2 is inside every rule's domain).  NO surrogate model, NO
MASTER, NO store-elite seeding is used inside the construction loop -- the
whole point is to measure how much of what 100-call search campaigns find is
captured by the explicit rules.  The champion model is used AFTERWARDS, by a
separate script, purely as a referee.

WHAT "FROM THE RULES ALONE" MEANS HERE
--------------------------------------
Hard gates (rules that are feasibility gates, applied as constraints):
  G1  centre slot = E2, the low-kinf0 / Gd-heavier fresh type
      (A::center-cold-fresh, Fisher p=8e-117; D::low-k-in-center).
  G2  no fresh fuel on the first axis orbit unit (quarter slots (0,1)/(1,0))
      (B::no-center-cross-fresh -- saturated to zero variance in-band).
  G3  structural invariants: twin symmetry, depth-1 acyclic chains, strict
      242-2F census -- enforced by ``GeneralOrbitGenome.validate()``
      (D::d-structural-negatives says enforce, never search).
  G4  net-INWARD burned shuffle, ``inward >= 0.75`` pitch
      (D::inward-migration-converges, AUC 0.681; in-band p2 = 0.77).
  G5  fr_hat = 1.047 * max(p_nom*FF(BU_nom)) <= 1.65 -- the family-C
      MAGNITUDE veto at its +-0.10 tolerance (never used to rank).
  G6  [flat profile only] the nominal hottest slot (46) carries fresh fuel --
      the E1/E2 role-split prerequisite (established fact 2; family C found
      92.5% of feasible rows obey it).
  G7  fresh-unit ring census fixed at (1, 4, 7, 18) units over the radial
      rings r<=2.5 / 2.5-4.5 / 4.5-6.5 / >6.5.  This is the family-A ring
      grammar (fresh-out-of-inner-ring, fresh-off-mid-crest,
      fresh-into-outer-band) read off at its in-band optimum: BOTH cell
      records (F_r 1.4636 and node_peak 1.1899) sit at exactly this census,
      and the feasible band clusters tightly around it (ring3 p50=p98=0.6923,
      ring2 p50 0.4118, ring0 p50 0.2381; derive_constants.py, n=381).
  G8  high-k (E1) unit count fixed by the boron budget rule
      (D::cbc-follows-k-budget + D::feed-split-serves-the-gates: kbud is
      pinned by the split in a depth-1 core; the measured feasible optimum is
      64 high-k assemblies of 121 for min-F_r, 68 for flat):
      ``minfr`` -> 16 E1 units, ``flat`` -> 17 E1 units.

Soft scoring (rules that are measured gradients, weighted by their validated
effect sizes; the greedy minimizes this score -- see ``score_minfr`` /
``score_flat``): RM1i down (B::rm1i-alive-verdict), FF radial gradient down
(A::ff-gradient-down, most portable), hot fresh outboard of cold
(A::hot-fresh-outboard), radial k-gradient / periphery-k up for flatness
(A::radial-k-gradient, A::periphery-k-mean -- E1_E2 is their home domain),
hot-ring fresh share HIGH for flat / moderate for minfr (B::hot-ring-fresh
two-regime elite flip; C::burned-peak-carrier endgame for minfr), burned-side
hot product up for minfr (C::burned-hot-product-style-direction), shuffle
travel short (D::short-shuffle-travel, +84 ppm CBC per pitch).  Metrics whose
validated in-band evidence is a TARGET rather than a direction (hb_fresh,
k_radgrad for minfr, travel, inward) enter as quadratic pulls toward the
in-band top-decile value, with envelope caps at the feasible-band p98 so the
constructor never extrapolates outside the band the rules were mined on
(the CBC co-budget warning of the validation verdict).

NOMINAL-MAP CONSTANTS: ``P_NOM_BOC`` / ``P_NOM_EOC`` are the cell-mean BOC /
EOC EDIT5 assembly-power planes over the 346 mapped feasible rows of this
cell (derive_constants.py, read-only store scan, 2026-08-11).  They are part
of the rule DEFINITIONS (families B and C define their metrics against the
cell's nominal map) -- they are labels of the CELL, not of any candidate.

ALGORITHM: seeded random census-satisfying start -> first-improvement greedy
descent over three closed move types (fresh/burned role swap within a ring,
E1/E2 batch swap, wiring source swap).  Deterministic given ``--seed``.
Every emitted construction passes ``GeneralOrbitGenome.validate()``,
``Pattern.validate_case``, and the pattern/record_id novelty check against
``data/store/records.parquet`` (plus any ``--exclude-json`` batches).

Usage (CPU-only, no MASTER, no model)::

    python rule_construct.py --profile minfr --n 200 --seed 1000 --out out.json
    python rule_construct.py --profile flat  --n 200 --seed 5000 --out out2.json \
        --exclude-json runs/v520/candidates.json

WRITES: the ``--out`` JSON only.  NEVER writes to data/store or data/models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from lpopt.data.flatness import (  # noqa: E402
    NEIGH_SLOT, NEIGH_VALID, PERIPHERY_MASK, SLOT_RADIUS, SLOT_WEIGHTS,
)
from lpopt.data.schema import compute_record_id, pack_pattern  # noqa: E402
from lpopt.search.construct import CAMPAIGN_DECK_KNOBS  # noqa: E402
from lpopt.search.genome import (  # noqa: E402
    GeneralOrbitGenome, GenomeError, ORBIT_UNITS,
)

# --------------------------------------------------------------------------- #
# cell fixture
# --------------------------------------------------------------------------- #
LIBRARY_ID = "ga80"
PAIR = "E1_E2"
FEED = 121
HOT_BATCH, COLD_BATCH = "E1", "E2"      # by kinf0 (1.1608 vs 1.0942, fuel_types)
CENTER_BATCH = COLD_BATCH               # gate G1
N_FRESH_UNITS = 30
#: gate G8 -- high-k fresh-unit count per profile (64 / 68 assemblies of 121).
N_HOT_UNITS = {"minfr": 16, "flat": 17}
#: family-A ring census (gate G7): fresh units per radial ring 0..3.
RING_CENSUS = (1, 4, 7, 18)
RING_EDGES = (2.5, 4.5, 6.5)
#: first axis orbit unit -- slots (0,1)/(1,0) -- gate G2.
C1_UNIT = 52
#: a-priori per-cycle burnup for (ga80, feed 121) [GWd/tU] (featurize regime).
B_REGIME = 28.69
#: family-A flat regime proxy for slot BOC burnup of an age-a assembly.
B_CYCLE = 22.0

#: Cell-mean nominal BOC/EOC assembly-power planes at the 69 quarter slots
#: (346 mapped feasible rows of ga80|E1_E2|f121; derive_constants.py).
P_NOM_BOC = np.array([
    1.11746, 1.06245, 1.17104, 1.14224, 1.12923, 1.15387, 1.18004, 1.10353,
    0.66434, 1.06245, 1.06485, 1.11262, 1.19106, 1.13643, 1.17736, 1.11522,
    1.10062, 0.66445, 1.17104, 1.14185, 1.18605, 1.14302, 1.17764, 1.10516,
    1.13153, 0.99210, 0.59661, 1.14224, 1.18810, 1.14902, 1.15098, 1.13092,
    1.12918, 1.09505, 0.88417, 0.42921, 1.12923, 1.10335, 1.12691, 1.17772,
    1.10408, 1.08332, 1.06052, 0.66237, 1.15387, 1.14551, 1.19182, 1.11500,
    1.14920, 1.12097, 0.86775, 0.41068, 1.18004, 1.14832, 1.13835, 1.18564,
    1.08086, 0.88410, 0.48283, 1.10353, 1.12222, 1.04880, 0.92048, 0.70671,
    0.44835, 0.66434, 0.64477, 0.60900, 0.43103])
P_NOM_EOC = np.array([
    1.12686, 0.89621, 1.16996, 0.96756, 0.92303, 1.00923, 1.09996, 1.09199,
    0.91762, 0.89621, 0.88626, 0.95443, 1.19438, 0.94747, 1.10056, 0.98045,
    1.20566, 0.93627, 1.16996, 0.97701, 1.19745, 0.97432, 1.17319, 0.97358,
    1.22405, 1.02921, 0.86183, 0.96756, 1.18368, 1.03157, 0.96497, 1.02314,
    1.11262, 1.12539, 1.07552, 0.66566, 0.92303, 0.90171, 0.92053, 1.13892,
    0.92274, 0.95440, 1.21242, 0.96794, 1.00923, 0.98507, 1.08304, 0.93484,
    1.10495, 1.10560, 1.09271, 0.64597, 1.09996, 0.96321, 1.02999, 1.11882,
    1.07369, 1.08667, 0.64142, 1.09199, 1.18871, 1.09629, 0.99650, 0.93671,
    0.69257, 0.91762, 0.84351, 0.84997, 0.60502])

#: Family-B nominal HOT RING: slots by descending P_NOM_BOC until the
#: multiplicity weight reaches 48 of 241 (weights sum exactly 48 here).
HOT_RING = (46, 12, 28, 20, 55, 6, 52, 39, 22, 14, 2, 18, 5, 44, 30)
#: gate G6 -- the nominal hottest slot.
HOTTEST_SLOT = int(np.argmax(P_NOM_BOC))            # == 46

#: In-band feasible envelope caps (p98 of the 381 feasible rows) -- the
#: constructor never pushes a dial beyond the band the rules were mined on.
CAP_K_RADGRAD = 0.72
CAP_TRAVEL = 2.30
GATE_INWARD_MIN = 0.75
GATE_FR_HAT_MAX = 1.65

W = SLOT_WEIGHTS
RAD = SLOT_RADIUS
INBOARD = ~PERIPHERY_MASK
RING_ID = np.digitize(RAD, RING_EDGES)               # 0..3 per slot
N_SLOTS = 69

#: unit -> (slots tuple, representative source slot, radius, ring)
UNIT_SLOTS = [u.slots for u in ORBIT_UNITS]
UNIT_REP = [u.slots[0] for u in ORBIT_UNITS]
UNIT_RADIUS = np.array([u.radius for u in ORBIT_UNITS])
UNIT_RING = np.digitize(UNIT_RADIUS, RING_EDGES)
UNIT_PNOM = np.array([
    float(sum(W[s] * P_NOM_BOC[s] for s in u.slots)
          / sum(W[s] for s in u.slots)) for u in ORBIT_UNITS])
UNIT_OF_SLOT = np.full(N_SLOTS, -1, dtype=int)
for _u, _def in enumerate(ORBIT_UNITS):
    for _s in _def.slots:
        UNIT_OF_SLOT[_s] = _u
HOTTEST_UNIT = int(UNIT_OF_SLOT[HOTTEST_SLOT])

_ring_check = tuple(int((UNIT_RING == r).sum()) for r in range(4))
assert _ring_check == (5, 12, 17, 26), _ring_check
assert UNIT_RING[C1_UNIT] == 0 and UNIT_RING[HOTTEST_UNIT] == 2


# --------------------------------------------------------------------------- #
# fuel physics (fuel_types.parquet, ga80 E1/E2) -- read once, read-only
# --------------------------------------------------------------------------- #
def load_fuel(store_dir: Path) -> dict:
    import pandas as pd

    ft = pd.read_parquet(store_dir / "fuel_types.parquet")
    ft = ft[ft["library_id"] == LIBRARY_ID].set_index("type_id")
    fuel = {}
    for t in (HOT_BATCH, COLD_BATCH):
        r = ft.loc[t]
        pts = [(0.0, float(r["kinf0"])), (10.0, float(r["kinf10"])),
               (20.0, float(r["kinf20"])), (30.0, float(r["kinf30"]))]
        if np.isfinite(float(r["kinf_eol50"])):
            pts.append((50.0, float(r["kinf_eol50"])))
        arr = np.asarray(pts)
        fuel[t] = {
            "kinf0": float(r["kinf0"]),
            "k22": float(np.interp(B_CYCLE, arr[:, 0], arr[:, 1])),
            "ff_max": float(r["ff_pin_max"]),
            "ff_r_inf": float(r["pin_bu_r_inf"]),
            "ff_paramA": float(r["pin_bu_paramA"]),
            "ff_asym": float(r["pin_bu_ratio_asym"]),
        }
    if fuel[HOT_BATCH]["kinf0"] <= fuel[COLD_BATCH]["kinf0"]:
        raise SystemExit("fuel table violates the E1(hot)/E2(cold) premise")
    return fuel


def _ff_at(fuel_t: dict, bu: float) -> float:
    """FF(BU) = clip(r_inf + paramA/BU, [ratio_asym, ff_pin_max]); FF(0)=max."""
    if bu <= 0.0:
        return fuel_t["ff_max"]
    return float(min(max(fuel_t["ff_r_inf"] + fuel_t["ff_paramA"] / bu,
                         fuel_t["ff_asym"]), fuel_t["ff_max"]))


# --------------------------------------------------------------------------- #
# state -> per-slot arrays -> rule metrics
# --------------------------------------------------------------------------- #
class State:
    """fresh: sorted list of 30 units; batch[unit]; wiring[burned]=source."""

    __slots__ = ("fresh", "batch", "wiring")

    def __init__(self, fresh, batch, wiring):
        self.fresh = fresh          # set[int]
        self.batch = batch          # dict[int, str] over fresh
        self.wiring = wiring        # dict[int, int] burned -> source (fresh)


def slot_arrays(st: State, fuel: dict):
    fresh = np.zeros(N_SLOTS, dtype=bool)
    hot = np.zeros(N_SLOTS, dtype=bool)         # origin batch == E1
    k = np.empty(N_SLOTS)
    ff = np.empty(N_SLOTS)
    src = np.full(N_SLOTS, -1, dtype=int)
    bu_nom = np.zeros(N_SLOTS)

    fresh[0] = True
    hot[0] = False
    k[0] = fuel[CENTER_BATCH]["kinf0"]
    ff[0] = fuel[CENTER_BATCH]["ff_max"]
    for u in st.fresh:
        b = st.batch[u]
        for s in UNIT_SLOTS[u]:
            fresh[s] = True
            hot[s] = b == HOT_BATCH
            k[s] = fuel[b]["kinf0"]
            ff[s] = fuel[b]["ff_max"]
    for burned, source in st.wiring.items():
        b = st.batch[source]
        rep = UNIT_REP[source]
        for s in UNIT_SLOTS[burned]:
            fresh[s] = False
            hot[s] = b == HOT_BATCH
            k[s] = fuel[b]["k22"]
            ff[s] = fuel[b]["ff_max"]
            src[s] = rep
            bu_nom[s] = B_REGIME * P_NOM_BOC[rep]
    return fresh, hot, k, ff, src, bu_nom


def _wcorr(x, y, w):
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0


def metrics(st: State, fuel: dict) -> dict:
    fresh, hot, k, ff, src, bu_nom = slot_arrays(st, fuel)

    m = {}
    m["k_radgrad"] = _wcorr(k, RAD, W)                       # A::radial-k-gradient
    m["k_outer"] = float(np.average(k[PERIPHERY_MASK],       # A::periphery-k-mean
                                    weights=W[PERIPHERY_MASK]))
    m["ff_radgrad"] = _wcorr(ff, RAD, W)                     # A::ff-gradient-down
    wh = W * fresh * hot
    wc = W * fresh * ~hot
    wc[0] = 0.0                                              # centre excluded
    m["hot_cold_dr"] = (float((wh * RAD).sum() / wh.sum())   # A::hot-fresh-outboard
                        - float((wc * RAD).sum() / wc.sum())) if wh.sum() and wc.sum() else 0.0
    nb = np.where(NEIGH_VALID, NEIGH_SLOT, 0)
    keep = (NEIGH_VALID & fresh[:, None] & fresh[nb]
            & INBOARD[:, None] & INBOARD[nb])
    m["rm1i"] = float(0.5 * (W * keep.sum(axis=1)).sum())    # B::rm1i-alive-verdict
    hr = np.asarray(HOT_RING)
    m["hb_fresh"] = float((W[hr] * fresh[hr]).sum()          # B::hot-ring-fresh
                          / W[hr].sum())
    burned = ~fresh
    dr = RAD[src[burned]] - RAD[burned]
    wb = W[burned]
    m["travel"] = float((wb * np.abs(dr)).sum() / (241 - FEED))   # D::short-shuffle-travel
    m["inward"] = float((wb * dr).sum() / (241 - FEED))           # D::inward-migration
    # family C nominal hot products (BOC + EOC trajectory)
    ffB = np.array([_ff_at(fuel[HOT_BATCH if h else COLD_BATCH], b)
                    for h, b in zip(hot, bu_nom)])
    ffE = np.array([_ff_at(fuel[HOT_BATCH if h else COLD_BATCH],
                           b + B_REGIME * pb)
                    for h, b, pb in zip(hot, bu_nom, P_NOM_BOC)])
    prodB = P_NOM_BOC * ffB
    prodE = P_NOM_EOC * ffE
    m["fr_hat"] = 1.047 * float(max(prodB.max(), prodE.max()))    # C veto (G5)
    m["hot_burned_nom"] = float(max(prodB[burned].max(),          # C style lever
                                    prodE[burned].max()))
    m["hottest_fresh"] = bool(fresh[HOTTEST_SLOT])
    m["nhot_units"] = int(sum(1 for u in st.fresh if st.batch[u] == HOT_BATCH))
    return m


# --------------------------------------------------------------------------- #
# rule scores (LOWER is better) -- weights = validated effect sizes
# --------------------------------------------------------------------------- #
def _env_pen(m: dict) -> float:
    """Envelope guard: stay inside the feasible band the rules were mined on."""
    p = 0.0
    if m["k_radgrad"] > CAP_K_RADGRAD:
        p += 5.0 * (m["k_radgrad"] - CAP_K_RADGRAD)
    if m["travel"] > CAP_TRAVEL:
        p += 0.5 * (m["travel"] - CAP_TRAVEL)
    if m["inward"] < GATE_INWARD_MIN:
        p += 10.0 * (GATE_INWARD_MIN - m["inward"])
    return p


def score_minfr(m: dict) -> float:
    """Pseudo-dF_r units.  Directional terms use the validated slopes:
    RM1i +0.0068/pair (elite), ff_radgrad +0.067/unit, hot_cold_dr -0.016/pitch,
    hot_burned_nom -0.24/unit (C4 -0.0062 per 0.026 IQR).  Target terms pull to
    the in-band low-F_r decile (hb_fresh 0.65, k_radgrad 0.63, travel 1.80,
    inward 1.35) -- B4's elite flip says do NOT maximize hot-ring fresh here;
    C1 hands the peak to burned fuel instead (hot_burned_nom up)."""
    s = 0.0
    s += 0.0068 * m["rm1i"]
    s += 0.067 * m["ff_radgrad"]
    s += -0.016 * m["hot_cold_dr"]
    s += -0.24 * min(max(m["hot_burned_nom"], 1.20), 1.36)
    s += 0.05 * ((m["hb_fresh"] - 0.65) / 0.10) ** 2
    s += 0.03 * ((m["k_radgrad"] - 0.63) / 0.15) ** 2
    s += 0.02 * ((m["travel"] - 1.80) / 0.30) ** 2
    s += 0.02 * ((m["inward"] - 1.35) / 0.30) ** 2
    return s + _env_pen(m)


def score_flat(m: dict) -> float:
    """Pseudo-dnode_peak units.  k-gradient axis with its measured slopes
    (-0.6/unit rho-scaled OLS; k_outer -3.0/unit, half the measured -6.7 for
    the collinearity with k_radgrad), RM1i +0.0028/pair (store slope), fresh
    anchors ON the hot ring (B4 elite regime, record hb 0.8333), small F_r
    protection via ff_radgrad and hot_cold_dr."""
    s = 0.0
    s += -0.60 * min(m["k_radgrad"], CAP_K_RADGRAD)
    s += -3.0 * (min(m["k_outer"], 1.1542) - 1.10)
    s += 0.0028 * m["rm1i"]
    # capped at the in-band maximum 0.8333 (the flat record's own value) --
    # beyond it the elite-regime evidence has no support (envelope discipline).
    s += -0.35 * min(m["hb_fresh"], 0.8333)
    s += 0.03 * m["ff_radgrad"]
    s += -0.005 * m["hot_cold_dr"]
    s += 0.02 * ((m["travel"] - 1.85) / 0.30) ** 2
    s += 0.02 * ((m["inward"] - 1.40) / 0.30) ** 2
    if not m["hottest_fresh"]:
        s += 1.0                       # gate G6 as a large penalty pre-reject
    return s + _env_pen(m)


SCORERS = {"minfr": score_minfr, "flat": score_flat}


# --------------------------------------------------------------------------- #
# construction: seeded start + first-improvement greedy descent
# --------------------------------------------------------------------------- #
def initial_state(profile: str, rng: random.Random) -> State:
    """Census-satisfying random start honouring gates G1/G2/G6/G7/G8."""
    fresh: set[int] = set()
    for ring, want in enumerate(RING_CENSUS):
        pool = [int(u) for u in np.flatnonzero(UNIT_RING == ring)
                if u != C1_UNIT]
        if profile == "flat" and ring == 2:
            fresh.add(HOTTEST_UNIT)                   # gate G6
            pool = [u for u in pool if u != HOTTEST_UNIT]
            fresh.update(rng.sample(pool, want - 1))
        else:
            fresh.update(rng.sample(pool, want))
    # batch split (gate G8): cold E2 on the highest-nominal-power fresh units
    # (established fact 2 role split), hot E1 on the rest.
    n_hot = N_HOT_UNITS[profile]
    order = sorted(fresh, key=lambda u: -UNIT_PNOM[u])
    batch = {}
    for i, u in enumerate(order):
        batch[u] = COLD_BATCH if i < (N_FRESH_UNITS - n_hot) else HOT_BATCH
    # wiring: greedy short-travel inward matching (D4/D6)
    burned = sorted((u for u in range(len(ORBIT_UNITS)) if u not in fresh),
                    key=lambda u: (-UNIT_RADIUS[u], u))
    free = set(fresh)
    wiring = {}
    for b in burned:
        best, best_c = None, None
        for s in sorted(free):
            c = abs(UNIT_RADIUS[s] - UNIT_RADIUS[b]) \
                + 1.5 * max(0.0, UNIT_RADIUS[b] - UNIT_RADIUS[s]) \
                + 0.001 * rng.random()
            if best_c is None or c < best_c:
                best, best_c = s, c
        wiring[b] = best
        free.discard(best)
    return State(fresh, batch, wiring)


def _moves(st: State, rng: random.Random, profile: str):
    """One randomly proposed closed move; yields a NEW State or None."""
    kind = rng.random()
    if kind < 0.4:
        # fresh/burned role swap within one ring (census-preserving)
        ring = rng.choice((0, 1, 2, 3))
        fr = [u for u in st.fresh if UNIT_RING[u] == ring]
        bu = [u for u in st.wiring if UNIT_RING[u] == ring and u != C1_UNIT]
        if profile == "flat":
            fr = [u for u in fr if u != HOTTEST_UNIT]
        if not fr or not bu:
            return None
        f = rng.choice(sorted(fr))
        b = rng.choice(sorted(bu))
        consumer = {s: d for d, s in st.wiring.items()}
        d = consumer[f]                       # f's consumer (perfect matching)
        wiring = dict(st.wiring)
        src_b = wiring.pop(b)                 # b becomes fresh
        if src_b == f:
            wiring[f] = b                     # edge reverses
        else:
            wiring[f] = src_b
            wiring[d] = b
        fresh = set(st.fresh)
        fresh.discard(f)
        fresh.add(b)
        batch = dict(st.batch)
        batch[b] = batch.pop(f)
        return State(fresh, batch, wiring)
    if kind < 0.65:
        # E1/E2 batch swap between two fresh units
        hots = sorted(u for u in st.fresh if st.batch[u] == HOT_BATCH)
        colds = sorted(u for u in st.fresh if st.batch[u] == COLD_BATCH)
        if not hots or not colds:
            return None
        h = rng.choice(hots)
        c = rng.choice(colds)
        batch = dict(st.batch)
        batch[h], batch[c] = batch[c], batch[h]
        return State(set(st.fresh), batch, dict(st.wiring))
    # wiring source swap between two burned units
    bs = sorted(st.wiring)
    if len(bs) < 2:
        return None
    b1, b2 = rng.sample(bs, 2)
    wiring = dict(st.wiring)
    wiring[b1], wiring[b2] = wiring[b2], wiring[b1]
    return State(set(st.fresh), dict(st.batch), wiring)


def construct_one(profile: str, seed: int, fuel: dict,
                  *, attempts_per_pass: int = 320, max_passes: int = 10):
    rng = random.Random(seed)
    st = initial_state(profile, rng)
    scorer = SCORERS[profile]
    best = scorer(metrics(st, fuel))
    for _ in range(max_passes):
        improved = False
        for _ in range(attempts_per_pass):
            cand = _moves(st, rng, profile)
            if cand is None:
                continue
            sc = scorer(metrics(cand, fuel))
            if sc < best - 1e-12:
                st, best, improved = cand, sc, True
        if not improved:
            break
    return st, best


def to_genome(st: State) -> GeneralOrbitGenome:
    genome = GeneralOrbitGenome(
        fresh=tuple(sorted((u, st.batch[u]) for u in st.fresh)),
        wiring=tuple(sorted(st.wiring.items())),
        center_batch=CENTER_BATCH,
    )
    genome.validate()
    return genome


# --------------------------------------------------------------------------- #
# novelty
# --------------------------------------------------------------------------- #
def _pattern_key(packed: str) -> str:
    return hashlib.blake2b(packed.encode("utf-8"), digest_size=16).hexdigest()


def load_taken(store_path: Path, exclude_json: list[Path]):
    import pandas as pd

    df = pd.read_parquet(store_path, columns=["record_id", "pattern"])
    keys = {_pattern_key(p) for p in df["pattern"].astype(str)}
    rids = set(df["record_id"].astype(str))
    n_store = len(df)
    for p in exclude_json:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        for c in payload.get("candidates", []):
            keys.add(_pattern_key(str(c["pattern"])))
            rids.add(str(c["record_id"]))
    return keys, rids, n_store


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="rule-only LP constructor "
                                             "(ga80 | E1_E2 | feed 121)")
    ap.add_argument("--profile", required=True, choices=("minfr", "flat"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000,
                    help="first seed of the ladder (seed, seed+1, ... until "
                         "--n distinct novel constructions are held)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="data/store/records.parquet")
    ap.add_argument("--exclude-json", action="append", default=[],
                    help="frozen candidates.json of an unmerged batch whose "
                         "patterns must be treated as already taken")
    ap.add_argument("--no-novelty", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    fuel = load_fuel(BASE / "data/store")
    taken_keys: set = set()
    taken_rids: set = set()
    n_store = 0
    if not args.no_novelty:
        taken_keys, taken_rids, n_store = load_taken(
            BASE / args.store, [BASE / e for e in args.exclude_json])
        print(f"novelty index: {n_store} store rows"
              f" + {len(args.exclude_json)} exclusion file(s)")

    held = []
    held_keys: set = set()
    seed = args.seed
    tried = 0
    dropped_dup = dropped_store = dropped_gate = 0
    while len(held) < args.n and tried < args.n * 20:
        tried += 1
        st, rule_score = construct_one(args.profile, seed, fuel)
        seed += 1
        try:
            genome = to_genome(st)
            pattern = genome.to_pattern()
            pattern.validate_case(PAIR, FEED)
        except GenomeError as exc:
            dropped_gate += 1
            print(f"  [seed {seed - 1}] genome invalid: {exc}")
            continue
        m = metrics(st, fuel)
        # hard gates G4/G5 (+G6 for flat) -- reject, do not repair
        if (m["inward"] < GATE_INWARD_MIN or m["fr_hat"] > GATE_FR_HAT_MAX
                or (args.profile == "flat" and not m["hottest_fresh"])):
            dropped_gate += 1
            continue
        packed = pack_pattern(pattern)
        key = _pattern_key(packed)
        rid = compute_record_id(pattern.canonical(), LIBRARY_ID, PAIR,
                                CAMPAIGN_DECK_KNOBS)
        if key in held_keys:
            dropped_dup += 1
            continue
        if key in taken_keys or rid in taken_rids:
            dropped_store += 1
            continue
        held_keys.add(key)
        held.append({
            "profile": args.profile,
            "seed": seed - 1,
            "record_id": rid,
            "record_id_preimage": {
                "canonical_pattern": packed,
                "library_id": LIBRARY_ID,
                "case_pair": PAIR,
                "deck_knobs": CAMPAIGN_DECK_KNOBS,
                "rule": "record_id = sha256(canonical|library|pair|knobs)",
            },
            "pattern": packed,
            "pattern_key": key,
            "rule_score": rule_score,
            "metrics": {k: (float(v) if not isinstance(v, bool) else bool(v))
                        for k, v in m.items()},
        })
        if len(held) % 50 == 0:
            print(f"  held {len(held)}/{args.n}  (seed {seed - 1}, "
                  f"{time.time() - t0:.0f}s)")

    if len(held) < args.n:
        raise SystemExit(f"only {len(held)} constructions from "
                         f"{tried} seeds -- widen the ladder")

    payload = {
        "constructor": "rule_construct.py (rules-only; no model, no MASTER)",
        "rules": "scratchpad rules/validated.json 2026-08-11 "
                 "(18 CONFIRMED + 6 DOMAIN-LIMITED)",
        "cell": {"library_id": LIBRARY_ID, "pair": PAIR, "feed": FEED},
        "profile": args.profile,
        "n": len(held),
        "seed_ladder_start": args.seed,
        "deck_knobs": CAMPAIGN_DECK_KNOBS,
        "novelty": {"checked": not args.no_novelty, "store_rows": n_store,
                    "dropped_in_store": dropped_store,
                    "dropped_duplicate": dropped_dup,
                    "dropped_gate": dropped_gate,
                    "exclusions": list(args.exclude_json)},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "constructions": held,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{len(held)} {args.profile} constructions -> {out}")
    print(f"  seeds consumed {args.seed}..{seed - 1}  "
          f"dropped: dup={dropped_dup} in_store={dropped_store} "
          f"gate={dropped_gate}  wall {time.time() - t0:.0f}s")
    rs = np.array([h["rule_score"] for h in held])
    print(f"  rule_score: min {rs.min():.4f} p50 {np.median(rs):.4f} "
          f"max {rs.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
