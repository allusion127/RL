"""``lpopt.tools.quarantine_campaign`` — dry-run default, scoped edit, backups.

The tool exists for labels that are converged and wrong (the HGD569 alias-no-op
defect, memo 20260830).  What is pinned here is the safety contract: nothing is
written without ``--apply``, exactly the named campaign's rows change, no row is
ever deleted from the store, and a second run is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lpopt.data.store import RECORDS_NAME, StoreWriter
from lpopt.tools.quarantine_campaign import main

CAMPAIGN = "intervention_HGD569_f125"
OTHER = "fpcamp_minfr_hgd569_f125"
FAILURE = "alias_noop_P6_20260830"


def _record(idx: int, campaign: str):
    from lpopt.data.schema import CanonicalRecord

    return CanonicalRecord(
        record_id=f"{idx:064x}", dataset="P", campaign=campaign, stratum=None,
        generator="intervention_1move", parent_record_id=None,
        case_pair="A_B", feed=125, n_batches=40, depth2_edges=0,
        e_core=5.69, e_split=0.1, library_id="paramA", sym_class="C1",
        pattern="F:A:0", f_r=1.6, f_q=2.0, cbc_max=1200.0, cbc_boc=1100.0,
        cbc_kind="max", cyclen=766.0, ao_abs=0.1, cycle_burnup=None,
        discharge_burnup=None, max_assembly_burnup=None, max_pin_burnup=None,
        eoc_ppm=None, delta_efpd=None, n_cycles=12.0, converged=True,
        converged_at_cap=False, tolerance_margin=0.1,
        restart_provenance="pair_ecore:MAS_RST.APRQ_11_0705.02",
        valid=True, failure="", maps_key=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    store_dir = tmp_path / "store"
    writer = StoreWriter(store_dir)
    writer.write_records(
        [_record(i, CAMPAIGN if i < 4 else OTHER) for i in range(6)],
        derive_enrichment_columns=False,
    )
    return store_dir


def _steps(tmp_path: Path) -> Path:
    path = tmp_path / "steps.parquet"
    pd.DataFrame({
        "campaign": [CAMPAIGN] * 3 + [OTHER] * 2,
        "parent_record_id": list("abcde"),
        "child_record_id": list("vwxyz"),
    }).to_parquet(path, index=False)
    return path


def _argv(store: Path, steps: Path, backups: Path, *extra: str) -> list[str]:
    return [
        "--campaign", CAMPAIGN, "--failure", FAILURE,
        "--store-dir", str(store), "--steps", str(steps),
        "--backup-dir", str(backups), "--backup-tag", "test", *extra,
    ]


def test_dry_run_writes_nothing(store: Path, tmp_path: Path) -> None:
    steps = _steps(tmp_path)
    backups = tmp_path / "backups"
    before = (store / RECORDS_NAME).read_bytes(), steps.read_bytes()

    assert main(_argv(store, steps, backups)) == 0

    assert (store / RECORDS_NAME).read_bytes() == before[0]
    assert steps.read_bytes() == before[1]
    assert not backups.exists()


def test_apply_scopes_the_edit_and_keeps_every_row(store: Path, tmp_path: Path) -> None:
    steps = _steps(tmp_path)
    backups = tmp_path / "backups"

    assert main(_argv(store, steps, backups, "--apply")) == 0

    df = pd.read_parquet(store / RECORDS_NAME)
    assert len(df) == 6, "quarantine must never delete a store row"
    hit = df[df["campaign"] == CAMPAIGN]
    rest = df[df["campaign"] == OTHER]
    assert len(hit) == 4 and not hit["valid"].any()
    assert set(hit["failure"]) == {FAILURE}
    # converged is an observed fact and is left alone without --unconverge
    assert hit["converged"].all()
    # every other campaign is untouched
    assert rest["valid"].all() and set(rest["failure"]) == {""}

    # steps: only this campaign's edges are dropped
    st = pd.read_parquet(steps)
    assert len(st) == 2 and set(st["campaign"]) == {OTHER}

    # both files were copied aside first
    assert (backups / f"{RECORDS_NAME}.bak_pre_test_"
            f"{pd.Timestamp.now():%Y%m%d}").exists()
    assert list(backups.glob("steps.parquet.bak_pre_test_*"))


def test_second_apply_is_a_noop(store: Path, tmp_path: Path) -> None:
    steps = _steps(tmp_path)
    backups = tmp_path / "backups"
    assert main(_argv(store, steps, backups, "--apply")) == 0
    after = (store / RECORDS_NAME).read_bytes(), steps.read_bytes()

    # the backup guard refuses to clobber, so a re-run needs a fresh dir
    assert main(_argv(store, steps, tmp_path / "backups2", "--apply")) == 0
    assert (store / RECORDS_NAME).read_bytes() == after[0]
    assert steps.read_bytes() == after[1]


def test_unconverge_escalates_rows_a_previous_run_quarantined(
        store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two-pass: quarantine first, escalate with ``--unconverge`` second.

    Regression -- the second pass used to look at the campaign's rows, see they
    were already ``valid=False`` with the same failure tag, and treat the whole
    campaign as settled, so ``--unconverge`` never cleared ``converged`` and the
    rows kept seeding elite pools and surrogate training.
    """

    steps = _steps(tmp_path)
    # pass 1 -- labels quarantined, converged deliberately left alone
    assert main(_argv(store, steps, tmp_path / "b1", "--apply")) == 0
    first = pd.read_parquet(store / RECORDS_NAME)
    assert first[first["campaign"] == CAMPAIGN]["converged"].all()

    # pass 2 -- same campaign, same failure tag, now with --unconverge
    assert main(_argv(store, steps, tmp_path / "b2", "--apply", "--unconverge")) == 0
    df = pd.read_parquet(store / RECORDS_NAME)
    hit = df[df["campaign"] == CAMPAIGN]
    assert len(df) == 6 and len(hit) == 4
    assert not hit["converged"].any(), "--unconverge must clear ALREADY-quarantined rows"
    assert not hit["valid"].any() and set(hit["failure"]) == {FAILURE}
    rest = df[df["campaign"] == OTHER]
    assert rest["valid"].all() and rest["converged"].all()

    # pass 3 -- nothing left to do: 0 rows to change, no bytes written, and no
    # backup attempted (a re-run must not die on the backup-exists guard).
    after = (store / RECORDS_NAME).read_bytes(), steps.read_bytes()
    capsys.readouterr()
    assert main(_argv(store, steps, tmp_path / "b3", "--apply", "--unconverge")) == 0
    out = capsys.readouterr().out
    assert "rows to change          0" in out
    assert (store / RECORDS_NAME).read_bytes() == after[0]
    assert steps.read_bytes() == after[1]
    assert not (tmp_path / "b3").exists()


def test_unconverge_is_opt_in(store: Path, tmp_path: Path) -> None:
    steps = _steps(tmp_path)
    assert main(_argv(store, steps, tmp_path / "b", "--apply", "--unconverge")) == 0
    df = pd.read_parquet(store / RECORDS_NAME)
    hit = df[df["campaign"] == CAMPAIGN]
    assert not hit["converged"].any() and not hit["valid"].any()
    assert df[df["campaign"] == OTHER]["converged"].all()
