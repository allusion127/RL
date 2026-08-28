"""Offline replay regression for the wave-selection fix (plan sec. 4.6).

This is the guard that would have caught the M5-pilot bug (run
``20260717_022611``): with the REAL store and the REAL trained ensemble it
builds one K1_K2 / feed-121 wave and asserts the exploit slots land in the
verified-feasible elite neighbourhood, not the high-σ OOD tail.

The three defects it pins:

1. ``_store_elites`` must return verified-*feasible* parents (feasibility-first),
   so the pool actually contains feasible-neighbourhood mutation children.  In
   the pilot 0/32 "elites" were feasible.
2. The exploit slots must rank on the risk-adjusted **exploit score**, not the
   exploration-weighted acquisition; the exploit picks must be the lowest
   predicted-F_r (most-feasible) candidates, not the maximum-σ ones.
3. The explore slots (and only they) chase raw epistemic σ.

MODEL-BIAS NOTE.  The plan brief's literal thresholds for this test — predicted
``F_r <= 1.7`` and predicted ``cyclen in [620, 670]`` — assume an unbiased
surrogate.  The current champion ensemble systematically **over-predicts** on
the verified-feasible K1_K2 basin: it maps the 70 feasible rows (measured
F_r 1.54, cyclen ~655) to predicted F_r ~1.83 and cyclen ~693.  So the
elite-neighbourhood, as the model sees it, is F_r ~1.79-1.81 / cyclen ~693, and
the assertions below are calibrated to that reality.  The absolute over-
prediction is a *model-quality* problem (out of scope for the selection fix);
the selection's job is to pick the feasible basin, which these assertions verify.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpopt.config import load_config
from lpopt.search import acquisition as acq
from lpopt.search.construct import build_pool

_ROOT = Path(__file__).resolve().parents[1]
_DECK = _ROOT / "lpopt.inp"
_STORE = _ROOT / "data" / "store"


def _ensemble_dir() -> Path | None:
    cfg_dir = _ROOT / "data" / "models"
    if not cfg_dir.exists():
        return None
    for d in sorted(cfg_dir.glob("*")):
        if any(d.glob("member_*")):
            return d
    return None


_HAVE = _DECK.exists() and (_STORE / "records.parquet").exists() and _ensemble_dir() is not None


@pytest.mark.skipif(not _HAVE, reason="real deck / store / trained ensemble not present")
def test_replay_exploit_picks_feasible_neighbourhood(tmp_path):
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.search.campaign import CampaignDriver
    from lpopt.search.stub import StubEvaluator

    cfg = load_config(_DECK)
    backend = PosValCnnBackend.from_dir(
        _ensemble_dir(), store_dir=str(_STORE), library_id=cfg.model.library_id
    )
    driver = CampaignDriver(
        cfg, backend, evaluator_factory=lambda w, c: StubEvaluator(),
        dry_run=True, run_dir=tmp_path / "run", budget=100, progress=False, seed=0,
    )
    assert driver.ctx.case_key.label == "K1_K2/feed-121"

    # -- 1. feasibility-first elite query -------------------------------------
    elites = driver._store_elites()
    assert len(elites) >= 8
    elite_ids = {rid for rid, _ in elites}
    df = driver._store_df()
    by_id = {str(r["record_id"]): r for _, r in df.iterrows()}

    def _feas(rid: str) -> bool:
        r = by_id.get(rid)
        if r is None:
            return False
        try:
            return bool(
                r["f_r"] <= cfg.acquisition.f_r_limit
                and r["cbc_max"] <= cfg.acquisition.cbc_limit
                and r["f_q"] <= cfg.acquisition.f_q_limit
                and r["ao_abs"] is not None
                and abs(float(r["ao_abs"])) <= cfg.acquisition.ao_abs_limit
            )
        except (TypeError, ValueError):
            return False

    feasible_elites = [rid for rid in elite_ids if _feas(rid)]
    # store holds 70 verified-feasible K1_K2 rows; the elite slots must be filled
    # by them (the pilot bug: 0/32 feasible).
    assert len(feasible_elites) >= 0.9 * len(elites)

    # -- 2. pool carries feasible-neighbourhood mutation children -------------
    pool = build_pool(
        driver.ctx, driver.model, elites, driver.ledger_ids, driver.rng,
        driver.cfg, wave_index=0, prev_top=[], size=driver.pool_size,
    )
    elite_children = [i for i, c in enumerate(pool) if c.parent_record_id in elite_ids]
    assert len(elite_children) >= 10                     # ~60% elite-mutation share

    boot = driver.model.predict(
        [c.pattern for c in pool], driver.ctx.case_key, driver.ctx.e_core or 0.0
    )
    rm = acq.build_reward_model(driver.ctx, [c.pattern for c in pool], boot, driver.constraints)
    scored = acq.score_pool(
        driver.model, driver.ctx, pool, rm, driver.constraints, driver.trust_region,
        have_feasible=False,
    )
    child_fr = scored.mean[elite_children, 0]
    # elite children sit in the feasible basin the model sees (~1.8), well below
    # the OOD tail (>2.2) — NOT the pilot's F_r 1.9-2.5 "elite" children.
    assert float(np.nanmin(child_fr)) <= 1.85

    # -- 3. run the full selection stack --------------------------------------
    scored = acq.local_search(
        driver.model, driver.ctx, scored, rm, driver.constraints, driver.trust_region,
        driver.local_search_cfg, driver.rng, driver.ledger_ids, have_feasible=False,
    )
    tau = acq.tau_schedule(scored, driver.acq.tau0, have_feasible=False)
    slots = acq.compose_wave(
        scored, [], driver.rng, size=cfg.acquisition.wave_size,
        n_exploit=cfg.acquisition.exploit, n_explore=cfg.acquisition.explore,
        n_control=cfg.acquisition.control, tau=tau, hamming_min=cfg.acquisition.hamming_min,
    )
    exploit_idx = [s.index for s in slots if s.slot == "exploit"]
    explore_idx = [s.index for s in slots if s.slot == "explore"]
    assert len(exploit_idx) == cfg.acquisition.exploit
    assert len(explore_idx) == cfg.acquisition.explore

    ex_fr = scored.mean[exploit_idx, 0]
    ex_cyc = scored.mean[exploit_idx, 3]
    xp_fr = scored.mean[explore_idx, 0]

    # (a) every exploit pick is in the feasible-basin neighbourhood, not OOD.
    #     Pilot exploit picks were F_r 1.89-2.03; the fixed floor is ~1.81.
    assert float(np.max(ex_fr)) <= 1.82
    #     cyclen in the elite neighbourhood (loose: the +40 EFPD model bias makes
    #     the plan's [620,670] unreachable with THIS champion — see module note).
    assert np.all((ex_cyc >= 620.0) & (ex_cyc <= 700.0))

    # (b) exploit strictly beats explore on feasibility (low F_r) ...
    assert float(np.max(ex_fr)) < float(np.min(xp_fr))
    # (c) ... while explore, and only explore, chases uncertainty (higher σ).
    assert scored.raw_epi[exploit_idx].mean() < scored.raw_epi[explore_idx].mean()

    # (d) direct anti-regression: ranking exploit on the exploit score is at
    #     least as feasible as ranking on the exploration acquisition (the bug).
    region = np.flatnonzero(scored.in_region)
    acq_top5 = region[np.argsort(-scored.acq[region])][: cfg.acquisition.exploit]
    assert ex_fr.mean() <= scored.mean[acq_top5, 0].mean() + 1e-6
