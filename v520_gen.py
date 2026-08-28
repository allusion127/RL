"""verify-5-of-20 (objection K2) -- CANDIDATE GENERATOR.  Costs ZERO MASTER calls.

Pre-registration: ``data/reports/v520_preregistration_20260810.md`` (written
2026-08-10, BEFORE this file existed).  Sections cited below are that document's.

WHAT THIS DOES (pre-registration section 2.2 / 2.3 / 2.4 / 2.5)

  1. Builds the campaign's own candidate pool for cell ga80 | E1_E2 | feed 121
     and scores it with the champion, by reproducing the ``min_fr_max_cycle``
     branch of ``CampaignDriver._proposals_only`` (``search/campaign.py``:2050-2120)
     VERBATIM -- same ``build_pool`` call, same ``acq.score_pool_min_fr`` call,
     same driver, same deck, same seed.

     THE ONE DEVIATION, and the reason it exists: ``_proposals_only`` truncates
     at ``np.argsort(-scored.rank)[:16]`` (campaign.py:2095).  **16 is a
     hard-coded literal, not a knob**, and this experiment needs 20 NOVEL ones.
     So the full ``ScoredPool`` is exported instead of the top 16.  Nothing about
     how candidates are BUILT or SCORED changes.

  2. Selection, in this order (section 2.3):
       (a) admissible  -- ``in_region`` and finite ``rank`` (the campaign's own
                          trust-region hard gate; out-of-region rows are -inf and
                          no exploit slot could ever adopt them)
       (b) novel       -- section 2.4, against the frozen store snapshot: packed
                          pattern absent store-wide AND campaign record_id absent
       (c) distinct    -- pairwise distinct packed patterns
       (d) cut         -- top N by ``scored.rank`` descending (the campaign's own
                          exploit preference order)
     Registered top-up: if fewer than N survive, re-draw with random_seed 4637,
     4638, ... in ascending order and append, recording each candidate's seed.

  3. Ranks the held N by PREDICTED MEAN F_r ASCENDING (``ScoredPool.mean[:, 0]``
     -- the ensemble mean, not the UCB the acquisition scalar uses), ties broken
     by ascending record_id, and FREEZES that ranking to
     ``runs/v520/candidates.json`` with its sha256 printed.  ``v520_run.py``
     consumes that file and never re-ranks.

WHY NOVELTY IS A HARD REQUIREMENT (section 2.4)
  Re-measuring a core the store already holds is not verification -- the search
  has already seen it, and its F_r is a lookup, not a measurement.  Two tests are
  applied because they are NOT equivalent: dataset A rows were identified with
  ``deck_knobs="mocha_default"`` and dataset B rows with ``"ga_native"``, so the
  same 69 cards can sit in the store under a record_id the campaign hash would
  never reproduce.  Pattern equality catches those; record_id equality catches an
  exact campaign-path duplicate.  Failing EITHER test drops the candidate.

Usage (no MASTER; safe to re-run)::

    python v520_gen.py                      # 20 candidates -> runs/v520/candidates.json
    python v520_gen.py --n 20 --force       # overwrite an existing frozen file

WRITES: ``runs/v520/candidates.json``, ``runs/v520/gen/`` (driver scratch).
NEVER writes to ``data/store/`` or ``data/models/``.
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

#: Frozen ranking + provenance consumed by ``v520_run.py``.  Section 2.5.
CANDIDATES_NAME = "candidates.json"

#: Seed ladder for the registered top-up rule (section 2.3).  4636 is the deck's
#: own ``[flow] random_seed`` (it names the incumbent F_r 1.4636); the rest are
#: consumed IN ORDER and only if the previous draw left the holding short.
DEFAULT_SEEDS = (4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643)

#: ``SurrogatePrediction`` 7-column contract (``model_api`` module docstring):
#: (F_r, CBC_max, F_q, cyclen, AO_abs, max_assembly_burnup, max_pin_burnup).
PRED_COLS = ("f_r", "cbc_max", "f_q", "cyclen", "ao_abs")
PRED_IDX = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# store novelty index
# --------------------------------------------------------------------------- #
def _pattern_key(packed: str) -> str:
    """Compact, collision-free-in-practice key for a packed pattern string."""
    return hashlib.blake2b(packed.encode("utf-8"), digest_size=16).hexdigest()


class StoreIndex:
    """The frozen novelty oracle (pre-registration section 2.4 / 3.1).

    Read-only.  Holds hashed packed patterns (store-wide -- deliberately the
    widest test, across every library / pair / feed) and the full record_id set,
    plus the converged-only subsets so the verdict can report both the strict
    gate and the weaker "already has a CONVERGED row" phrasing of gate V4.
    """

    def __init__(self, store_path: Path):
        import pandas as pd

        if not store_path.is_file():
            raise SystemExit(f"store not found: {store_path}")
        raw = store_path.read_bytes()
        self.path = store_path
        self.n_bytes = len(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        del raw

        df = pd.read_parquet(store_path, columns=["record_id", "pattern", "converged"])
        self.n_rows = int(len(df))
        conv = df["converged"] == True  # noqa: E712
        self.n_converged = int(conv.sum())
        self.pattern_keys = {_pattern_key(p) for p in df["pattern"].astype(str)}
        self.record_ids = set(df["record_id"].astype(str))
        self.conv_pattern_keys = {
            _pattern_key(p) for p in df.loc[conv, "pattern"].astype(str)
        }
        self.conv_record_ids = set(df.loc[conv, "record_id"].astype(str))

        #: Hand-added exclusions (:meth:`exclude_candidates`) -- patterns that are
        #: NOT in the store yet but must be treated as if they were.
        self.extra_pattern_keys: set[str] = set()
        self.extra_record_ids: set[str] = set()
        self.extra_sources: list[dict] = []

    def exclude_candidates(self, path: Path) -> dict:
        """Treat another batch's frozen ``candidates.json`` as already-taken.

        REQUIRED whenever a previous batch's MASTER labels have not yet been
        merged into ``records.parquet``: the store cannot exclude what it does
        not hold, and two batches sharing a pattern would not be independent
        draws.  Recorded in the output so the verdict can state whether the
        exclusion was load-bearing (``store_hits`` 0 means it WAS: the store
        would not have caught them).
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        cands = payload.get("candidates") or []
        store_hits = 0
        for c in cands:
            packed = str(c["pattern"])
            rid = str(c["record_id"])
            if _pattern_key(packed) in self.pattern_keys or rid in self.record_ids:
                store_hits += 1
            self.extra_pattern_keys.add(_pattern_key(packed))
            self.extra_record_ids.add(rid)
        info = {"path": str(path), "n": len(cands), "already_in_store": store_hits,
                "load_bearing": store_hits < len(cands)}
        self.extra_sources.append(info)
        return info

    def hit(self, packed: str, record_id: str) -> tuple[bool, bool]:
        """``(in_store_any, in_store_converged)`` for one candidate."""
        k = _pattern_key(packed)
        any_hit = (k in self.pattern_keys or record_id in self.record_ids
                   or k in self.extra_pattern_keys or record_id in self.extra_record_ids)
        conv_hit = k in self.conv_pattern_keys or record_id in self.conv_record_ids
        return any_hit, conv_hit


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_driver(deck: Path, run_dir: Path, *, quiet: bool = False):
    """The champion + the campaign driver, exactly as ``cli.cmd_optimize`` builds them.

    ``budget=0`` is the proposals-only contract (``campaign.py``:1461-1462): no
    evaluator is ever constructed for a MASTER call on this path.  The driver is
    still built with ``evaluator_factory=None`` (the live path) so that the
    trust region, the elites, the constraints and the ``MinFrSpec`` are the exact
    objects a real wave would score with -- an evaluator that is never invoked
    changes none of them.
    """
    from lpopt.config import load_config
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.search.campaign import CampaignDriver

    cfg = load_config(deck)
    cfg.case.validate()
    cfg.constraints.validate()

    deck_dir = deck.resolve().parent
    store_dir = deck_dir / cfg.model.store_dir
    model_dir = deck_dir / cfg.model.model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"model dir not found: {model_dir}")

    model = PosValCnnBackend.from_dir(
        model_dir, store_dir=store_dir,
        library_id=cfg.model.library_id, device=cfg.model.device,
    )
    log = (lambda m: None) if quiet else (lambda m: print(f"    {m}"))
    driver = CampaignDriver(
        cfg, model, None, budget=0, run_dir=run_dir, progress=False, log=log,
    )
    if driver.objective != "min_fr_max_cycle":
        raise SystemExit(
            f"deck objective is {driver.objective!r}; the pre-registration fixes "
            "min_fr_max_cycle (v520_minfr_local.inp)")
    return cfg, model, driver


def draw_pool(driver, seed: int):
    """One pool draw + score -- the ``min_fr_max_cycle`` body of ``_proposals_only``.

    Re-seeding ``driver.rng`` is equivalent to constructing the driver with
    ``seed=<seed>`` for this code path: ``build_pool`` is the only consumer of
    the rng between construction and scoring, and ``_store_elites()`` is a
    deterministic store read.
    """
    from lpopt.search import acquisition as acq
    from lpopt.search.construct import build_pool

    driver.rng = random.Random(int(seed))
    pool = build_pool(
        driver.ctx, driver.model, driver._store_elites(), driver.ledger_ids,
        driver.rng, driver.cfg, wave_index=0, size=driver.pool_size,
    )
    scored = acq.score_pool_min_fr(
        driver.model, driver.ctx, pool, driver.min_fr_spec, driver.trust_region,
        tie_epsilon=float(driver.acq.tie_epsilon),
    )
    return scored


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def select_from_scored(scored, store: StoreIndex, *, seed: int,
                       held_keys: set[str], want: int) -> tuple[list[dict], dict]:
    """Steps (a)-(d) of pre-registration section 2.3 on ONE scored pool.

    Returns ``(entries, counts)``; ``entries`` is already cut to ``want`` and
    ordered by ``rank`` descending.  ``counts`` is the audit trail the verdict
    quotes as proof the novelty gate actually ran.
    """
    from lpopt.data.schema import pack_pattern

    n = len(scored)
    rank = np.asarray(scored.rank, dtype=float)
    region = np.asarray(scored.in_region, dtype=bool)
    mean = np.asarray(scored.mean, dtype=float)

    counts = {
        "seed": int(seed), "pool": int(n),
        "admissible": 0, "novel": 0, "distinct": 0, "taken": 0,
        "dropped_out_of_region": 0, "dropped_nonfinite_rank": 0,
        "dropped_in_store_any": 0, "dropped_in_store_converged": 0,
        "dropped_duplicate_in_pool": 0, "dropped_already_held": 0,
    }

    rows: list[dict] = []
    seen_this_pool: set[str] = set()
    # rank DESCENDING == the campaign's own exploit preference order, the key
    # `_proposals_only` prints and an exploit slot adopts.
    for i in np.argsort(-rank, kind="stable"):
        i = int(i)
        if not region[i]:
            counts["dropped_out_of_region"] += 1
            continue
        if not np.isfinite(rank[i]):
            counts["dropped_nonfinite_rank"] += 1
            continue
        counts["admissible"] += 1

        cand = scored.candidates[i]
        packed = pack_pattern(cand.pattern)
        key = _pattern_key(packed)
        any_hit, conv_hit = store.hit(packed, cand.record_id)
        if any_hit:
            counts["dropped_in_store_any"] += 1
            if conv_hit:
                counts["dropped_in_store_converged"] += 1
            continue
        counts["novel"] += 1

        if key in seen_this_pool:
            counts["dropped_duplicate_in_pool"] += 1
            continue
        seen_this_pool.add(key)
        if key in held_keys:
            counts["dropped_already_held"] += 1
            continue
        counts["distinct"] += 1

        if len(rows) >= want:
            continue
        rows.append({
            "record_id": str(cand.record_id),
            "pattern": packed,
            "pattern_key": key,
            "origin": str(cand.origin),
            "parent_record_id": (None if cand.parent_record_id is None
                                 else str(cand.parent_record_id)),
            "seed": int(seed),
            "pool_index": i,
            "exploit_rank_key": float(rank[i]),
            "exploit": float(scored.exploit[i]),
            "margin": float(scored.margin[i]),
            "p_feas": float(scored.p_feas[i]),
            "acq": float(scored.acq[i]),
            "pred": {c: float(mean[i, j]) for c, j in zip(PRED_COLS, PRED_IDX)},
            "pred_epistemic_f_r": float(np.asarray(scored.epistemic)[i, 0]),
            "pred_calibrated_f_r": float(np.asarray(scored.calibrated)[i, 0]),
            "shf": cand.pattern.to_shf(),
        })
    counts["taken"] = len(rows)
    return rows, counts


def freeze_ranking(entries: list[dict]) -> list[dict]:
    """Pre-registration section 2.5: predicted MEAN F_r ascending, record_id tie-break."""
    ordered = sorted(entries, key=lambda e: (e["pred"]["f_r"], e["record_id"]))
    for k, e in enumerate(ordered, start=1):
        e["predicted_rank"] = k
    return ordered


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="verify-5-of-20 candidate generator (budget-0; NO MASTER)")
    ap.add_argument("--input", "-i", default="v520_minfr_local.inp",
                    help="LOCAL-path deck (never fpcamp_minfr_199.inp)")
    ap.add_argument("--n", type=int, default=20, help="candidates to freeze (pre-reg: 20)")
    ap.add_argument("--run-dir", default="runs/v520")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="registered seed ladder, consumed in order, only as needed")
    ap.add_argument("--exclude-json", action="append", default=[],
                    help="another batch's frozen candidates.json whose patterns must "
                         "be treated as already-taken.  REQUIRED when that batch's "
                         "labels have not yet been merged into records.parquet -- the "
                         "store cannot exclude what it does not hold, and two batches "
                         "sharing a pattern are not independent draws.  Repeatable.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen candidates.json")
    ap.add_argument("--quiet-driver", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE))

    deck = (BASE / args.input).resolve()
    if not deck.is_file():
        raise SystemExit(f"deck not found: {deck}")
    if deck.name == "fpcamp_minfr_199.inp":
        raise SystemExit("refusing to run against the 199 deck -- use v520_minfr_local.inp")

    run_dir = (BASE / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / CANDIDATES_NAME
    if out_path.is_file() and not args.force:
        raise SystemExit(
            f"{out_path} already exists -- the frozen ranking must not be silently "
            "replaced (pre-registration section 5.2 gate V5).  Pass --force only if "
            "NO MASTER chain has run against it.")

    print("=" * 78)
    print("verify-5-of-20 CANDIDATE GENERATION -- budget 0, ZERO MASTER calls")
    print("pre-registration: data/reports/v520_preregistration_20260810.md")
    print("=" * 78)

    store = StoreIndex(BASE / "data/store/records.parquet")
    print(f"store            : {store.n_rows} rows ({store.n_converged} converged), "
          f"{store.n_bytes} bytes")
    print(f"store sha256     : {store.sha256}")
    for ex in args.exclude_json:
        p = (BASE / ex).resolve()
        if not p.is_file():
            raise SystemExit(f"--exclude-json not found: {p}")
        info = store.exclude_candidates(p)
        print(f"hand-exclusion   : {p.name}  n={info['n']}  "
              f"already_in_store={info['already_in_store']}  "
              f"LOAD-BEARING={info['load_bearing']}"
              + ("  <- the store would NOT have caught these"
                 if info["load_bearing"] else ""))

    t0 = time.time()
    print(f"deck             : {deck.name}")
    cfg, model, driver = build_driver(deck, run_dir / "gen", quiet=args.quiet_driver)
    print(f"case             : {driver.ctx.case_key.label}  e_core={driver.ctx.e_core}")
    print(f"objective        : {driver.objective}   pool_size={driver.pool_size}   "
          f"tie_epsilon={driver.acq.tie_epsilon}")
    print(f"model            : {driver.champion_ckpt}")
    print(f"driver built in  : {time.time() - t0:.1f}s")

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    held: list[dict] = []
    held_keys: set[str] = set()
    audit: list[dict] = []

    for seed in seeds:
        want = args.n - len(held)
        if want <= 0:
            break
        print(f"\n-- pool draw seed={seed}  (need {want} more) " + "-" * 30)
        t1 = time.time()
        scored = draw_pool(driver, seed)
        rows, counts = select_from_scored(
            scored, store, seed=seed, held_keys=held_keys, want=want)
        counts["wall_s"] = round(time.time() - t1, 1)
        audit.append(counts)
        for r in rows:
            held.append(r)
            held_keys.add(r["pattern_key"])
        print(f"   pool={counts['pool']}  admissible={counts['admissible']}  "
              f"novel={counts['novel']}  distinct={counts['distinct']}  "
              f"taken={counts['taken']}  ({counts['wall_s']}s)")
        print(f"   dropped: out_of_region={counts['dropped_out_of_region']}  "
              f"in_store={counts['dropped_in_store_any']} "
              f"(converged {counts['dropped_in_store_converged']})  "
              f"dup_in_pool={counts['dropped_duplicate_in_pool']}  "
              f"already_held={counts['dropped_already_held']}")
        print(f"   held so far: {len(held)} / {args.n}")

    if len(held) < args.n:
        raise SystemExit(
            f"only {len(held)} novel distinct candidates from seeds {seeds} -- "
            f"extend --seeds (the ladder is registered as ascending) and re-run")

    ordered = freeze_ranking(held)

    origins: dict[str, int] = {}
    by_seed: dict[str, int] = {}
    for e in ordered:
        origins[e["origin"]] = origins.get(e["origin"], 0) + 1
        by_seed[str(e["seed"])] = by_seed.get(str(e["seed"]), 0) + 1

    fr = [e["pred"]["f_r"] for e in ordered]
    payload = {
        "experiment": "v520 verify-5-of-20 (objection K2)",
        "preregistration": "data/reports/v520_preregistration_20260810.md",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deck": deck.name,
        "case": {"pair": driver.ctx.pair, "feed": int(driver.ctx.feed),
                 "e_core": driver.ctx.e_core, "library_id": driver.library_id},
        "objective": driver.objective,
        "model_dir": driver.champion_ckpt,
        "cond_schema": cfg.model.cond_schema,
        "pool_size": int(driver.pool_size),
        "seeds_used": [int(s) for s in by_seed],
        "seed_ladder": seeds,
        "selection_rule": ("admissible(in_region & finite rank) -> novel(store: "
                           "packed-pattern OR campaign record_id) -> distinct("
                           "packed pattern) -> top-N by ScoredPool.rank desc"),
        "ranking_rule": "predicted MEAN F_r ascending (mean[:,0]), record_id tie-break",
        "store": {"path": str(store.path), "rows": store.n_rows,
                  "converged": store.n_converged, "bytes": store.n_bytes,
                  "sha256": store.sha256},
        "hand_exclusions": store.extra_sources,
        "counts": {"n": len(ordered), "origins": origins, "by_seed": by_seed,
                   "predicted_f_r_min": min(fr), "predicted_f_r_max": max(fr),
                   "predicted_f_r_range": max(fr) - min(fr)},
        "audit": audit,
        "candidates": ordered,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print("\n" + "=" * 78)
    print(f"FROZEN {len(ordered)} candidates -> {out_path}")
    print(f"candidates.json sha256 = {sha}")
    print("=" * 78)
    print(f"{'rk':>3} {'record_id':<14}{'origin':<10}{'seed':>6}"
          f"{'pred F_r':>10}{'pred cyc':>10}{'pred CBC':>10}{'pred F_q':>9}"
          f"{'p_feas':>8}{'rank':>12}")
    for e in ordered:
        p = e["pred"]
        print(f"{e['predicted_rank']:>3} {e['record_id'][:12]:<14}{e['origin']:<10}"
              f"{e['seed']:>6}{p['f_r']:>10.4f}{p['cyclen']:>10.2f}{p['cbc_max']:>10.1f}"
              f"{p['f_q']:>9.4f}{e['p_feas']:>8.3f}{e['exploit_rank_key']:>12.2f}")
    print(f"\npredicted F_r range: {min(fr):.4f} .. {max(fr):.4f} "
          f"(spread {max(fr) - min(fr):.4f})")
    print(f"origins: {origins}   by seed: {by_seed}")
    print("\nNOVELTY PROOF (pre-registration section 2.4 / gate V4):")
    print(f"  all {len(ordered)} packed patterns absent from the {store.n_rows}-row store")
    print(f"  all {len(ordered)} campaign record_ids absent from the store")
    for ex in store.extra_sources:
        print(f"  all {len(ordered)} disjoint from {Path(ex['path']).name} "
              f"({ex['n']} patterns hand-excluded; {ex['already_in_store']} of them "
              f"were in the store, so the exclusion "
              f"{'WAS' if ex['load_bearing'] else 'was not'} load-bearing)")
    print(f"  {len(ordered)} pairwise-distinct packed patterns "
          f"({len({e['pattern_key'] for e in ordered})} unique keys)")
    print("\nNO MASTER was launched.  Next: run_v520.bat (the ISSUER launches it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
