"""``lpopt.tools.repair_parent_ids`` — the one-off dangling-foreign-key pass.

The tool has exactly three verdicts per non-null ``parent_record_id``, and each
is registered here:

1. **resolves** — already a store row; never touched;
2. **re-derivable** — the parent BOARD is in the store under a different cell and
   the child re-keyed it into its own; the true id is recovered from the
   cell-invariant ``Pattern.digest`` (this is the cross-cell donor shape);
3. **phantom** — the board is in the store under NO key, so nothing can be
   pointed at.  Left alone by default; ``--null-phantom`` clears the false claim.

Plus the write contract: dry-run touches nothing, ``--apply`` backs up first,
preserves row order, and a second run is a no-op.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from lpopt.data.schema import (
    SYM_CLASS, CanonicalRecord, compute_record_id, pack_pattern,
)
from lpopt.data.store import RECORDS_NAME, StoreWriter
from lpopt.search.genome import random_genome
from lpopt.search.verify import PRODUCE_DECK_KNOBS
from lpopt.tools import repair_parent_ids as rp

HOT = "P6253Z1G06N24"
MID = "P6253Z2G10N20"
COLD = "P6253Z2G10N24"
DONOR = f"{HOT}_{COLD}"
TRIPLE = f"{HOT}_{MID}_{COLD}"
FEED = 125
N_FRESH = (FEED - 1) // 4
LIB = "paramA"
PHANTOM = "f" * 64          # a 64-hex id no board in the store can produce


def _pat(seed: int):
    return random_genome(random.Random(seed), DONOR, N_FRESH).to_pattern()


def _rec(pattern, *, pair: str, campaign: str, parent: str | None,
         generator: str = "local") -> CanonicalRecord:
    return CanonicalRecord(
        record_id=compute_record_id(
            pattern.canonical(), LIB, pair, PRODUCE_DECK_KNOBS),
        dataset="P", campaign=campaign, stratum="min_fr", generator=generator,
        parent_record_id=parent, case_pair=pair, feed=FEED, n_batches=2,
        depth2_edges=0, e_core=5.69, e_split=0.18, library_id=LIB,
        sym_class=SYM_CLASS, pattern=pack_pattern(pattern),
        f_r=1.55, f_q=2.0, cbc_max=1500.0, cbc_boc=None, cbc_kind="max",
        cyclen=730.0, ao_abs=0.03, cycle_burnup=None, discharge_burnup=None,
        max_assembly_burnup=None, max_pin_burnup=None, eoc_ppm=None,
        delta_efpd=None, n_cycles=11.0, converged=True, converged_at_cap=False,
        tolerance_margin=0.2, restart_provenance="x", valid=True, failure="",
        maps_key=None,
    )


@pytest.fixture
def store(tmp_path: Path):
    """A store with one donor-cell board and three TRIPLE children of it.

    The children are: one with the donor's TRUE id (resolves), one with the
    donor board RE-KEYED into the TRIPLE cell (re-derivable), and one naming a
    board that does not exist anywhere (phantom).
    """

    d = tmp_path / "store"
    d.mkdir(parents=True, exist_ok=True)

    donor_pat = _pat(1)
    donor = _rec(donor_pat, pair=DONOR, campaign="donor_camp", parent=None,
                 generator="elite")
    rekeyed = compute_record_id(
        donor_pat.canonical(), LIB, TRIPLE, PRODUCE_DECK_KNOBS)
    assert rekeyed != donor.record_id

    kids = [
        _rec(_pat(10), pair=TRIPLE, campaign="triple", parent=donor.record_id),
        _rec(_pat(11), pair=TRIPLE, campaign="triple", parent=rekeyed),
        _rec(_pat(12), pair=TRIPLE, campaign="triple", parent=PHANTOM),
    ]
    StoreWriter(d).write_records([donor, *kids])
    return d, donor, kids


def _scan(store_dir: Path):
    df = pd.read_parquet(store_dir / RECORDS_NAME)
    return rp.scan(df, campaigns=["triple"], min_rows=1)


def test_verdicts_resolve_rederive_and_phantom(store):
    d, donor, kids = store
    reports, edits = _scan(d)
    rep = reports["triple"]
    assert (rep.non_null, rep.resolved, rep.repaired, rep.phantom) == (3, 1, 1, 1)
    # the re-keyed child is repaired to the donor's TRUE (donor-cell) id...
    assert edits[kids[1].record_id] == donor.record_id
    # ...and the phantom is staged as a null, never as a guess.
    assert edits[kids[2].record_id] is None
    # the already-good child is not an edit at all.
    assert kids[0].record_id not in edits


def test_dry_run_writes_nothing(store):
    d, _, _ = store
    before = (d / RECORDS_NAME).read_bytes()
    assert rp.main(["--store-dir", str(d), "--campaign", "triple",
                    "--min-rows", "1", "--null-phantom"]) == 0
    assert (d / RECORDS_NAME).read_bytes() == before
    assert not list(d.glob("*.bak_pre_parentid_*"))


def test_apply_backs_up_preserves_order_and_is_idempotent(store):
    d, donor, kids = store
    path = d / RECORDS_NAME
    order = pd.read_parquet(path)["record_id"].tolist()

    assert rp.main(["--store-dir", str(d), "--campaign", "triple",
                    "--min-rows", "1", "--null-phantom", "--apply"]) == 0
    backups = list(d.glob("*.bak_pre_parentid_*"))
    assert len(backups) == 1

    after = pd.read_parquet(path).set_index("record_id")
    assert pd.read_parquet(path)["record_id"].tolist() == order   # order intact
    assert after.loc[kids[0].record_id, "parent_record_id"] == donor.record_id
    assert after.loc[kids[1].record_id, "parent_record_id"] == donor.record_id
    assert pd.isna(after.loc[kids[2].record_id, "parent_record_id"])

    # second pass: every remaining parent resolves, so nothing is staged and the
    # file is not rewritten (a second backup would also have raised).
    reports, edits = _scan(d)
    assert not edits
    assert reports["triple"].phantom == 0
    assert reports["triple"].resolved == 2


def test_phantom_is_left_alone_without_the_flag(store):
    d, _, kids = store
    assert rp.main(["--store-dir", str(d), "--campaign", "triple",
                    "--min-rows", "1", "--apply"]) == 0
    after = pd.read_parquet(d / RECORDS_NAME).set_index("record_id")
    # repaired, but the phantom claim is preserved for the operator to decide on.
    assert after.loc[kids[2].record_id, "parent_record_id"] == PHANTOM
