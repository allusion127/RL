"""``lpopt.tools.backfill_fxy`` — final-cycle adjudication, digest join, apply.

Two contracts are pinned here.  (1) The SCAN must refuse every work dir it cannot
prove is the final equilibrium cycle: a local runs tree is mostly other cycles
that outlived a failed Windows ``rmtree`` (design 20260829 §2.1/§5.3), and a
mid-chain FXYP recorded as an equilibrium label is silent data poisoning.  (2)
The APPLY must never overwrite, never guess, and never lose the parquet — it is
the same idempotent / atomic / order-preserving / never-destructive contract
``backfill_flatness`` carries, plus a backup, because this pass's source tree can
live on another box.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from lpopt.data.schema import CanonicalRecord, pack_pattern, unpack_pattern
from lpopt.data.store import (
    RECORDS_NAME, StoreWriter, ensure_schema_columns, frame_to_table,
    records_to_frame,
)
from lpopt.search.genome import random_genome
from lpopt.tools.backfill_fxy import (
    apply, digest_from_deck, digest_of_packed, read_csv, scan, write_csv,
)

from tests.test_fxy import FIXTURE


def _pattern(seed: int):
    return random_genome(random.Random(seed), "K1_K2", 30).to_pattern()


def _packed(seed: int) -> str:
    """A real 69-cell packed pattern (the store's ``pattern`` column shape)."""
    return pack_pattern(_pattern(seed))


PACKED_A = _packed(1)
PACKED_B = _packed(2)


# --------------------------------------------------------------------------- #
# synthetic runs tree
# --------------------------------------------------------------------------- #
def _work_dir(parent: Path, digest: str, tag: str, *, job: int,
              restart: str, mas_sum: bool = True, nonfinite: bool = False,
              fxyp: str = "1.8247", suffix: str = "-0123456789-abcdefgh") -> Path:
    """One staged MASTER work dir, named exactly as ``verify.py:931`` +
    ``master.py``'s ``mkdtemp`` prefix build it."""
    work = parent / f"{digest}__{tag}{suffix}"
    work.mkdir(parents=True)
    (work / "MAS_OUT").write_text(FIXTURE.replace("1.8247", fxyp), encoding="ascii")
    (work / "MAS_INP").write_text(
        f"%LPD_RST\n        {restart}\n\n%JOB_IDE\n        APRQ    {job}\n",
        encoding="ascii")
    if mas_sum:
        (work / "MAS_SUM").write_text("SUMMARY EDIT 2 : REACTIVITY\n", encoding="ascii")
    if nonfinite:
        (work / "NONFINITE_FLUX").write_text("", encoding="ascii")
    return work


@pytest.fixture()
def runs_tree(tmp_path: Path) -> tuple[Path, str, str]:
    """A runs tree with one dir of each adjudication class."""
    root = tmp_path / "runs"
    worker = root / "camp" / "master" / "master_work" / "worker_00"
    worker.mkdir(parents=True)
    seed = "MAS_RST.APRQ_10_0615.11"
    tag = "MAS_RST.APRQ_10_0615.1"        # the 40-char truncation of ``seed``

    dig_ok = "a" * 16
    dig_first = "b" * 16
    dig_nf = "c" * 16
    dig_nosum = "d" * 16
    dig_sup = "e" * 16

    _work_dir(worker, dig_ok, tag, job=12, restart="MAS_RST.APRQ_11_0619.10")
    _work_dir(worker, dig_first, tag, job=2, restart=seed)          # cycle 1
    _work_dir(worker, dig_nf, tag, job=12, restart="MAS_RST.APRQ_11_0619.10",
              nonfinite=True)
    _work_dir(worker, dig_nosum, tag, job=12, restart="MAS_RST.APRQ_11_0619.10",
              mas_sum=False)
    # two dirs of ONE chain: only the higher %JOB_IDE is the final cycle.
    _work_dir(worker, dig_sup, tag, job=5, restart="MAS_RST.APRQ_04_0619.10",
              fxyp="1.9999", suffix="-1111111111-aaaaaaaa")
    _work_dir(worker, dig_sup, tag, job=12, restart="MAS_RST.APRQ_11_0619.10",
              suffix="-2222222222-bbbbbbbb")
    # not a staged case dir at all (no digest prefix).
    stray = worker / "scratch"
    stray.mkdir()
    (stray / "MAS_OUT").write_text(FIXTURE, encoding="ascii")
    (stray / "MAS_SUM").write_text("x", encoding="ascii")
    return root, dig_ok, dig_sup


# --------------------------------------------------------------------------- #
# synthetic ARCHIVE trees (2_LP/LOW_Fr_MASTER_result shapes)
# --------------------------------------------------------------------------- #
#: the two archive layouts the collection sweep produced, reproduced exactly:
#: a flattened one-directory-per-case origin with a ``manifest.csv`` beside it,
#: and ``regen/<id>/cyNN`` chains with a ``_meta_chain.json`` per chain.
MANIFEST_HEADER = ("origin,case_name,case_path,chain_class,n_cycles,"
                   "cyc_min,cyc_max\n")


def _deck(seed: int | None, *, restart: str, job: int) -> str:
    """A MASTER reload deck whose ``%LPD_SHF`` body IS pattern ``seed``.

    ``seed=None`` writes a deck with no parsable shuffle body (the bootstrap
    ``irrst=0`` shape), which is what forces the name fallback.
    """
    head = (f"%LPD_RST\n        {restart}\n\n"
            f"%JOB_IDE\n        APRQ    {job}\n\n")
    if seed is None:
        return head + "%LPD_SHF                       # ishuff\n        0 stead\n%END\n"
    return (head + "%LPD_SHF                       # ishuff\n"
            + _pattern(seed).to_shf(final_newline=True) + "%END\n")


def _case(parent: Path, name: str, *, seed: int | None = 1,
          restart: str = "MAS_RST.APRQ_11_0619.10", job: int = 12,
          mas_sum: bool = True, fxyp: str = "1.8247") -> Path:
    """One collected work dir: a deck, a MAS_OUT, and (usually) a MAS_SUM."""
    work = parent / name
    work.mkdir(parents=True)
    (work / "MAS_INP").write_text(_deck(seed, restart=restart, job=job),
                                  encoding="ascii")
    (work / "MAS_OUT").write_text(FIXTURE.replace("1.8247", fxyp),
                                  encoding="ascii")
    if mas_sum:
        (work / "MAS_SUM").write_text("SUMMARY EDIT 2 : REACTIVITY\n",
                                      encoding="ascii")
    return work


def _manifest(archive: Path, rows: list[tuple[str, ...]]) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "manifest.csv").write_text(
        MANIFEST_HEADER + "".join(",".join(r) + "\n" for r in rows),
        encoding="utf-8")


@pytest.fixture()
def flat_archive(tmp_path: Path) -> tuple[Path, str]:
    """A flattened origin: names ``~``-elided, keys recoverable only from decks.

    Four cases from ONE original worker directory, mirroring what the sweep did
    to ``srv181``: the staged ``<digest16>__<tag>`` prefix is gone from every
    on-disk name and each cycle of a chain became its own top-level case.
    """
    archive = tmp_path / "LOW_Fr"
    origin = archive / "srv_x"
    tag = "MAS_RST.APRQ_10_0615.1"
    worker = "C:/Users/USER/lpopt_work/camp/master_work/worker_03"
    dig = digest_of_packed(PACKED_A)

    # (on-disk flattened name, original dir name, seed, restart, job)
    plan = [
        ("camp_master_work_worker_03~RST.APRQ_10_0615.1-aaaaaaaaaa-fin_1111aaaa",
         f"{dig}__{tag}-aaaaaaaaaa-final000", 1, "MAS_RST.APRQ_11_0619.10", 12),
        # cycle 1 of the same chain: its deck still reads the SEED restart.
        ("camp_master_work_worker_03~RST.APRQ_10_0615.1-bbbbbbbbbb-fir_2222bbbb",
         f"{digest_of_packed(PACKED_B)}__{tag}-bbbbbbbbbb-first000", 2,
         "MAS_RST.APRQ_10_0615.11", 2),
        # an earlier cycle of the FIRST chain: same digest+tag, lower %JOB_IDE.
        ("camp_master_work_worker_03~RST.APRQ_10_0615.1-cccccccccc-sup_3333cccc",
         f"{dig}__{tag}-cccccccccc-super000", 1, "MAS_RST.APRQ_04_0619.10", 5),
    ]
    for name, orig, seed, restart, job in plan:
        _case(origin, name, seed=seed, restart=restart, job=job)
    _manifest(archive, [
        ("srv_x", name, f"{worker}/{orig}", "final_cycle_only", "1", "", "")
        for name, orig, _s, _r, _j in plan
    ])
    return origin, dig


# --------------------------------------------------------------------------- #
# scan / adjudication
# --------------------------------------------------------------------------- #
def test_scan_adjudicates_each_class(runs_tree) -> None:
    root, dig_ok, dig_sup = runs_tree
    rows, rep = scan(root)
    by_digest = {r.digest16: r for r in rows if r.cycle_evidence == "final"}
    verdicts = {r.cycle_evidence for r in rows}

    assert rep.n_dirs == 7
    assert verdicts == {"final", "first_cycle", "nonfinite", "no_mas_sum",
                        "superseded", "no_digest"}
    # exactly two finals: the clean dir and the LATER of the two chain siblings.
    assert set(by_digest) == {dig_ok, dig_sup}
    assert by_digest[dig_ok].f_xy == pytest.approx(1.8247)
    assert by_digest[dig_ok].f_xya == pytest.approx(1.6253)
    assert by_digest[dig_ok].n_steps == 3 and by_digest[dig_ok].sane is True
    assert by_digest[dig_ok].efpd_max == pytest.approx(15.0)
    # the superseded sibling's 1.9999 must NOT be what the chain reports.
    assert by_digest[dig_sup].f_xy == pytest.approx(1.8247)
    # skipped dirs are still reported (auditable), with no value.
    assert all(r.f_xy is None for r in rows if r.cycle_evidence != "final")
    assert rep.evidence["final"] == 2 and rep.n_sane == 2


def test_scan_flags_but_does_not_drop_garbage(tmp_path: Path) -> None:
    worker = tmp_path / "runs" / "c" / "w"
    worker.mkdir(parents=True)
    _work_dir(worker, "f" * 16, "MAS_RST.APRQ_10_0615.1", job=12,
              restart="MAS_RST.APRQ_11_0619.10", fxyp="5.1656")
    rows, rep = scan(tmp_path / "runs")
    assert rows[0].cycle_evidence == "final"        # it IS the final cycle ...
    assert rows[0].sane is False                    # ... but the value is garbage
    assert rows[0].reason == "above_ceiling" and rep.n_sane == 0


def test_scan_csv_round_trip(runs_tree, tmp_path: Path) -> None:
    root, dig_ok, _ = runs_tree
    rows, _rep = scan(root)
    out = write_csv(rows, tmp_path / "sub" / "scan.csv")
    back = read_csv(out)
    assert len(back) == len(rows)
    hit = [r for r in back if r["digest16"] == dig_ok][0]
    assert hit["cycle_evidence"] == "final" and hit["sane"] == "1"
    assert float(hit["f_xy"]) == pytest.approx(1.8247)
    skipped = [r for r in back if r["cycle_evidence"] == "nonfinite"][0]
    assert skipped["f_xy"] == "" and skipped["sane"] == "0"


# --------------------------------------------------------------------------- #
# archive: flattened case names
# --------------------------------------------------------------------------- #
def test_flat_archive_recovers_key_from_the_deck_and_reuses_the_rules(
        flat_archive) -> None:
    """The flattened names carry NO join key, so it comes from the deck — and
    ``manifest.csv``'s ``case_path`` puts the original name and worker directory
    back, so ``first_cycle`` / ``superseded`` adjudicate exactly as they do live.
    """
    origin, dig = flat_archive
    rows, rep = scan(origin, log=lambda m: None)

    assert rep.n_dirs == 3 and rep.n_manifest_rows == 3
    verdicts = {r.cycle_evidence for r in rows}
    assert verdicts == {"final", "first_cycle", "superseded"}
    final = [r for r in rows if r.cycle_evidence == "final"]
    assert len(final) == 1
    # the key is the pattern digest, rebuilt from %LPD_SHF, not read off a name.
    assert final[0].digest16 == dig and final[0].key_source == "deck"
    assert "__" not in Path(final[0].work_dir).name
    assert final[0].f_xy == pytest.approx(1.8247) and final[0].sane is True
    assert rep.key_sources == {"deck": 1} and rep.n_sane == 1
    # the superseded sibling still has a key recorded (auditable), no value.
    sup = [r for r in rows if r.cycle_evidence == "superseded"][0]
    assert sup.digest16 == dig and sup.f_xy is None


def test_flat_archive_falls_back_to_a_digest_in_the_name(tmp_path: Path) -> None:
    """When the deck will not parse, a ``[0-9a-f]{16}__`` ANYWHERE in the name is
    the last-resort key — position 0 is not assumed."""
    archive = tmp_path / "LOW_Fr"
    origin = archive / "srv_y"
    dig = digest_of_packed(PACKED_A)
    name = f"camp_worker_00_{dig}__MAS_RST.APRQ_11_0615.8_ab12cd34"
    _case(origin, name, seed=None)               # unparsable bootstrap deck
    _manifest(archive, [("srv_y", name, "C:/w/x/y", "final_cycle_only", "1",
                         "", "")])

    rows, rep = scan(origin, log=lambda m: None)
    assert len(rows) == 1
    assert rows[0].digest16 == dig and rows[0].key_source == "name"
    assert rows[0].cycle_evidence == "manifest:final_cycle_only"
    assert rows[0].sane is True and rep.key_sources == {"name": 1}


def test_flat_archive_manifest_final_cycle_only(tmp_path: Path) -> None:
    """A single-work-dir case the archive classifies ``final_cycle_only`` is the
    harvest-then-purge survivor; anything else is refused, not guessed."""
    archive = tmp_path / "LOW_Fr"
    origin = archive / "srv_z"
    _case(origin, "runs_camp_a~zzz_1111aaaa")
    _case(origin, "runs_camp_b~zzz_2222bbbb")
    _case(origin, "runs_camp_c~zzz_3333cccc")
    _manifest(archive, [
        ("srv_z", "runs_camp_a~zzz_1111aaaa", "C:/w/a", "final_cycle_only",
         "1", "", ""),
        ("srv_z", "runs_camp_b~zzz_2222bbbb", "C:/w/b", "restart_chain",
         "4", "2", "5"),
        # no manifest row at all for runs_camp_c.
    ])
    rows = {Path(r.work_dir).name: r for r in scan(origin, log=lambda m: None)[0]}
    assert rows["runs_camp_a~zzz_1111aaaa"].cycle_evidence == \
        "manifest:final_cycle_only"
    assert rows["runs_camp_b~zzz_2222bbbb"].cycle_evidence == "ambiguous_chain"
    assert rows["runs_camp_c~zzz_3333cccc"].cycle_evidence == "no_manifest_row"
    # refused rows still carry the recovered key, and never a value.
    assert all(r.key_source == "deck" for r in rows.values())
    assert all(r.f_xy is None for r in rows.values()
               if r.cycle_evidence != "manifest:final_cycle_only")


# --------------------------------------------------------------------------- #
# archive: cyNN chains
# --------------------------------------------------------------------------- #
def _regen_chain(archive: Path, chain_id: str, cycles: int, *,
                 converged: bool = True, last_sum: bool = True) -> Path:
    chain = archive / "regen" / chain_id
    chain.mkdir(parents=True)
    (chain / "_meta_chain.json").write_text(
        json.dumps({"record_id": "r" * 64, "pattern": PACKED_A,
                    "n_cycles": cycles, "converged": converged}),
        encoding="utf-8")
    for n in range(1, cycles + 1):
        _case(chain, f"cy{n:02d}", seed=1, job=n,
              mas_sum=(last_sum or n != cycles),
              fxyp="1.8247" if n == cycles else "1.9999")
    return chain


def test_regen_chain_harvests_only_the_highest_complete_cycle(
        tmp_path: Path) -> None:
    archive = tmp_path / "LOW_Fr"
    _regen_chain(archive, "00ecd022ec21", 4)
    rows, rep = scan(archive / "regen", log=lambda m: None)

    by_name = {Path(r.work_dir).name: r for r in rows}
    assert by_name["cy04"].cycle_evidence == "regen:cy04/4"
    assert [by_name[f"cy{n:02d}"].cycle_evidence for n in (1, 2, 3)] == \
        ["superseded"] * 3
    assert by_name["cy04"].digest16 == digest_of_packed(PACKED_A)
    assert by_name["cy04"].key_source == "deck"
    # the 1.9999 of the earlier cycles must not be what the chain reports.
    assert by_name["cy04"].f_xy == pytest.approx(1.8247)
    assert rep.n_sane == 1 and rep.evidence["regen"] == 1
    assert rep.evidence["superseded"] == 3      # bucketed, not one key per chain


def test_regen_chain_backs_off_to_the_last_complete_cycle(tmp_path: Path) -> None:
    """A trailing cycle that never wrote its MAS_SUM is a failed cycle dir; the
    chain's label is the highest cycle that has BOTH files."""
    archive = tmp_path / "LOW_Fr"
    _regen_chain(archive, "011d71e95aaf", 3, last_sum=False)
    rows = {Path(r.work_dir).name: r
            for r in scan(archive / "regen", log=lambda m: None)[0]}
    assert rows["cy03"].cycle_evidence == "no_mas_sum"
    assert rows["cy02"].cycle_evidence == "regen:cy02/3"


def test_regen_chain_that_did_not_converge_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "LOW_Fr"
    _regen_chain(archive, "019ba6251c4a", 3, converged=False)
    rows, rep = scan(archive / "regen", log=lambda m: None)
    assert {r.cycle_evidence for r in rows} == {"not_converged"}
    assert rep.n_sane == 0 and all(r.f_xy is None for r in rows)


def test_chain_case_pins_the_final_cycle_to_the_manifest_cyc_max(
        tmp_path: Path) -> None:
    """A ``local_3GA`` equilibrium case: ``full_cy11..13`` plus a parallel
    ``quarter_cy99`` safety branch.  ``cyc_max`` says the chain ends at 13, so
    the 99 cannot pose as its final cycle."""
    archive = tmp_path / "LOW_Fr"
    origin = archive / "local_3GA"
    case = origin / "tmp_safety_check~equilibrium_d541ef70"
    for n in (11, 12, 13):
        _case(case, f"full_cy{n:02d}", seed=1, job=n,
              fxyp="1.8247" if n == 13 else "1.9999")
    _case(case, "quarter_cy99", seed=2, job=99, fxyp="2.4000")
    _manifest(archive, [("local_3GA", case.name,
                         "C:/GA/tmp/safety_check/equilibrium",
                         "restart_chain", "4", "11", "13")])

    rows = {Path(r.work_dir).name: r for r in scan(origin, log=lambda m: None)[0]}
    assert rows["full_cy13"].cycle_evidence == "chain:cy13/4"
    assert rows["full_cy11"].cycle_evidence == "superseded"
    assert rows["quarter_cy99"].cycle_evidence == "superseded"
    assert rows["full_cy13"].f_xy == pytest.approx(1.8247)


def test_chain_rule_refuses_a_case_that_mixes_cycle_and_non_cycle_dirs(
        tmp_path: Path) -> None:
    """``bootstrap_work`` holds ``cy1`` beside unnumbered ``master__bootstrap-*``
    work dirs — there "the highest cyNN" is NOT the chain's end, so the whole
    case is refused rather than mislabelled."""
    archive = tmp_path / "LOW_Fr"
    origin = archive / "local_5RL"
    case = origin / "data_design_package_bootstrap_work_T5_T6_f81_49737dde"
    _case(case, "cy1", seed=1, job=1)
    _case(case, "master__bootstrap-0c7ad16153-9iwa7hli", seed=1, job=3)
    _manifest(archive, [("local_5RL", case.name, "C:/5RL/bootstrap_work/T5_T6_f81",
                         "full_chain_cy1", "3", "1", "1")])

    rows = scan(origin, log=lambda m: None)[0]
    assert {r.cycle_evidence for r in rows} == {"ambiguous_chain"}
    assert all(r.f_xy is None for r in rows)


def test_a_live_tree_is_untouched_by_the_archive_paths(runs_tree) -> None:
    """No ``manifest.csv``, no ``_meta_chain.json`` -> the original verdicts and
    the original name-derived key, unchanged."""
    root, dig_ok, dig_sup = runs_tree
    rows, rep = scan(root, log=lambda m: None)
    assert rep.n_manifest_rows == 0
    assert rep.key_sources == {"name": 2}
    assert {r.key_source for r in rows if r.cycle_evidence == "final"} == {"name"}
    assert sorted(r.digest16 for r in rows if r.cycle_evidence == "final") == \
        sorted([dig_ok, dig_sup])


# --------------------------------------------------------------------------- #
# digest join key
# --------------------------------------------------------------------------- #
def test_digest_from_deck_equals_the_store_digest(tmp_path: Path) -> None:
    """The deck-recomputed key must be the SAME 16 hex characters the store's
    packed ``pattern`` hashes to — otherwise the archive joins to nothing."""
    work = _case(tmp_path / "w", "flattened~name_deadbeef", seed=1)
    assert digest_from_deck(work) == digest_of_packed(PACKED_A)
    assert digest_from_deck(work) == unpack_pattern(PACKED_A).digest
    other = _case(tmp_path / "w2", "flattened~name_cafef00d", seed=2)
    assert digest_from_deck(other) == digest_of_packed(PACKED_B)


def test_digest_from_deck_returns_none_instead_of_raising(tmp_path: Path) -> None:
    assert digest_from_deck(tmp_path / "missing") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "MAS_INP").write_text("nothing here\n", encoding="ascii")
    assert digest_from_deck(empty) is None
    assert digest_from_deck(_case(tmp_path / "b", "bootstrap", seed=None)) is None


def test_digest_of_packed_equals_pattern_digest() -> None:
    """The fast path must be bit-identical to ``unpack_pattern(p).digest``."""
    for packed in (PACKED_A, PACKED_B):
        assert digest_of_packed(packed) == unpack_pattern(packed).digest
    assert digest_of_packed(PACKED_A) != digest_of_packed(PACKED_B)


def test_digest_of_packed_matches_a_real_store_row() -> None:
    path = Path("data/store") / RECORDS_NAME
    if not path.is_file():
        pytest.skip("no local store")
    head = pd.read_parquet(path, columns=["pattern"]).head(20)
    for packed in head["pattern"].astype(str):
        assert digest_of_packed(packed) == unpack_pattern(packed).digest


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def _record(rid: str, packed: str, **kw) -> CanonicalRecord:
    base = dict(
        record_id=rid, dataset="P", campaign="camp", stratum=None, generator="g",
        parent_record_id=None, case_pair="B1_C2", feed=121, n_batches=2,
        depth2_edges=0, e_core=5.4, e_split=0.1, library_id="260624",
        sym_class="rot61", pattern=packed, f_r=1.70, f_q=2.10, cbc_max=1500.0,
        cbc_boc=1480.0, cbc_kind="max", cyclen=15.0, ao_abs=0.05,
        cycle_burnup=27.0, discharge_burnup=54.0, max_assembly_burnup=67.0,
        max_pin_burnup=71.0, eoc_ppm=10.0, delta_efpd=0.5, n_cycles=11.0,
        converged=True, converged_at_cap=False, tolerance_margin=None,
        restart_provenance="mocha_native", valid=True, failure="", maps_key=None,
    )
    base.update(kw)
    return CanonicalRecord(**base)


def _store(tmp_path: Path, records: list[CanonicalRecord]) -> Path:
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True)
    StoreWriter(store_dir).write_records(records)
    return store_dir


def _csv(tmp_path: Path, packed: str, *, f_xy: float = 1.8247,
         f_xya: float = 1.6253, efpd: float = 15.0, evidence: str = "final",
         sane: str = "1", name: str = "scan.csv") -> Path:
    path = tmp_path / name
    path.write_text(
        "digest16,work_dir,cycle_evidence,f_xy,f_xya,n_steps,sane,reason,efpd_max\n"
        f"{digest_of_packed(packed)},wd,{evidence},{f_xy},{f_xya},25,{sane},,{efpd}\n",
        encoding="utf-8")
    return path


def test_apply_fills_by_digest_and_is_idempotent(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A), _record("r2", PACKED_B)])
    csv_path = _csv(tmp_path, PACKED_A)

    dry = apply(csv_path, store_dir, dry_run=True)
    assert dry.n_populated == 1 and dry.wrote is False and dry.backup == ""
    untouched = pd.read_parquet(store_dir / RECORDS_NAME)
    assert "f_xy" not in untouched or untouched["f_xy"].isna().all()

    rep = apply(csv_path, store_dir)
    assert rep.n_populated == 1 and rep.wrote is True
    assert rep.per_campaign == {"camp": 1}
    df = ensure_schema_columns(pd.read_parquet(store_dir / RECORDS_NAME))
    row = df.set_index("record_id")
    assert row.loc["r1", "f_xy"] == pytest.approx(1.8247)
    assert row.loc["r1", "f_xya"] == pytest.approx(1.6253)
    assert pd.isna(row.loc["r2", "f_xy"])          # the other pattern is untouched
    assert list(df["record_id"]) == ["r1", "r2"]   # order preserved

    again = apply(csv_path, store_dir)
    assert again.n_populated == 0 and again.n_already == 1 and again.wrote is False


def test_apply_never_overwrites_a_populated_cell(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A, f_xy=1.5000)])
    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    assert rep.n_already == 1 and rep.n_populated == 0 and rep.wrote is False
    df = pd.read_parquet(store_dir / RECORDS_NAME)
    assert df.loc[0, "f_xy"] == pytest.approx(1.5000)


def test_apply_backs_up_the_parquet_before_writing(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A)])
    before = (store_dir / RECORDS_NAME).read_bytes()
    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    backup = Path(rep.backup)
    assert backup.name.startswith(f"{RECORDS_NAME}.bak_pre_fxy_backfill_")
    assert backup.read_bytes() == before          # the PRE-write bytes
    assert pd.read_parquet(backup)["f_xy"].isna().all()


def test_apply_refuses_ambiguous_and_unjoinable_digests(tmp_path: Path) -> None:
    # same pattern, two records (different case pair): pattern-only key is
    # ambiguous, so nothing is written.
    store_dir = _store(tmp_path, [_record("r1", PACKED_A),
                                  _record("r2", PACKED_A, case_pair="D1_E2")])
    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    assert rep.n_ambiguous == 1 and rep.n_populated == 0 and rep.wrote is False

    # a digest no store row carries.
    store_dir2 = _store(tmp_path / "b", [_record("r1", PACKED_B)])
    rep2 = apply(_csv(tmp_path, PACKED_A, name="s2.csv"), store_dir2)
    assert rep2.n_no_store_row == 1 and rep2.n_populated == 0


def test_apply_skips_non_final_and_insane_csv_rows(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A)])
    for kw in ({"evidence": "first_cycle"}, {"evidence": "nonfinite"},
               {"evidence": "superseded"}, {"evidence": "ambiguous_chain"},
               {"evidence": "not_converged"}, {"evidence": "no_manifest_row"},
               {"sane": "0"}):
        rep = apply(_csv(tmp_path, PACKED_A, name="s.csv", **kw), store_dir)
        assert rep.n_final == 0 and rep.n_populated == 0 and rep.wrote is False


def test_apply_accepts_the_archive_final_cycle_evidences(tmp_path: Path) -> None:
    """``regen:cy12/12`` / ``chain:cy13/4`` / ``manifest:final_cycle_only`` are
    final-cycle verdicts carrying their own evidence; ``apply`` must join them
    exactly like a live ``final``."""
    for i, evidence in enumerate(("regen:cy12/12", "chain:cy13/4",
                                  "manifest:final_cycle_only")):
        store_dir = _store(tmp_path / f"e{i}", [_record("r1", PACKED_A)])
        rep = apply(_csv(tmp_path, PACKED_A, name=f"e{i}.csv",
                         evidence=evidence), store_dir)
        assert rep.n_final == 1 and rep.n_populated == 1 and rep.wrote is True


def test_apply_reads_the_key_source_split(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A)])
    digest = digest_of_packed(PACKED_A)
    csv_path = tmp_path / "ks.csv"
    csv_path.write_text(
        "digest16,work_dir,cycle_evidence,f_xy,f_xya,n_steps,sane,reason,"
        "efpd_max,key_source\n"
        f"{digest},a,regen:cy12/12,1.8247,1.6,25,1,,15.0,deck\n",
        encoding="utf-8")
    rep = apply(csv_path, store_dir, dry_run=True)
    assert rep.key_sources == {"deck": 1} and rep.n_populated == 1


def test_apply_refuses_a_wrong_cycle_and_a_broken_inequality(tmp_path: Path) -> None:
    # efpd_max disagrees with the record's cyclen -> this MAS_OUT is some OTHER
    # cycle of the chain, whatever the dir-level adjudication said.
    store_dir = _store(tmp_path, [_record("r1", PACKED_A, cyclen=619.098)])
    rep = apply(_csv(tmp_path, PACKED_A, efpd=15.0), store_dir)
    assert rep.n_cycle_mismatch == 1 and rep.n_populated == 0

    # F_r <= F_xy <= F_q is physics; a violation means contamination, not a label.
    store_dir2 = _store(tmp_path / "c", [_record("r1", PACKED_A, f_r=1.95)])
    rep2 = apply(_csv(tmp_path, PACKED_A, name="s3.csv"), store_dir2)
    assert rep2.n_inequality == 1 and rep2.n_populated == 0

    store_dir3 = _store(tmp_path / "d", [_record("r1", PACKED_A, f_q=1.60)])
    rep3 = apply(_csv(tmp_path, PACKED_A, name="s4.csv"), store_dir3)
    assert rep3.n_inequality == 1 and rep3.n_populated == 0


def test_apply_refuses_two_finals_that_disagree(tmp_path: Path) -> None:
    store_dir = _store(tmp_path, [_record("r1", PACKED_A)])
    digest = digest_of_packed(PACKED_A)
    csv_path = tmp_path / "dup.csv"
    csv_path.write_text(
        "digest16,work_dir,cycle_evidence,f_xy,f_xya,n_steps,sane,reason,efpd_max\n"
        f"{digest},a,final,1.8247,1.6,25,1,,15.0\n"
        f"{digest},b,final,1.7000,1.6,25,1,,15.0\n", encoding="utf-8")
    rep = apply(csv_path, store_dir)
    assert rep.n_dup_digest == 1 and rep.n_populated == 0 and rep.wrote is False


def test_apply_tolerates_a_parquet_written_before_the_columns(tmp_path: Path) -> None:
    """An old ``records.parquet`` has no ``f_xy`` column at all; it must still
    load, join and gain the column (the LATE_COLUMNS migration contract)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lpopt.data.schema import FROZEN_COLUMNS, PARQUET_SCHEMA
    from lpopt.data.store import records_to_frame

    store_dir = tmp_path / "old"
    store_dir.mkdir(parents=True)
    df = records_to_frame([_record("r1", PACKED_A)])[list(FROZEN_COLUMNS)]
    legacy = pa.schema([f for f in PARQUET_SCHEMA if f.name in FROZEN_COLUMNS])
    pq.write_table(pa.Table.from_pandas(df, schema=legacy, preserve_index=False),
                   store_dir / RECORDS_NAME)
    assert "f_xy" not in pd.read_parquet(store_dir / RECORDS_NAME).columns

    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    assert rep.n_populated == 1 and rep.wrote is True
    out = pd.read_parquet(store_dir / RECORDS_NAME)
    assert out.loc[0, "f_xy"] == pytest.approx(1.8247)


def test_apply_leaves_a_duplicated_record_id_alone_if_any_copy_is_filled(
        tmp_path: Path) -> None:
    """The store can hold two rows for one ``record_id``.  If ANY copy carries a
    value the record counts as filled — half-filling would leave one record with
    two different f_xy values on disk."""
    store_dir = tmp_path / "dup"
    store_dir.mkdir(parents=True)
    frame = records_to_frame([_record("r1", PACKED_A, f_xy=1.5000),
                              _record("r1", PACKED_A)])
    pq.write_table(frame_to_table(frame), store_dir / RECORDS_NAME)

    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    assert rep.n_already == 1 and rep.n_populated == 0 and rep.wrote is False
    out = pd.read_parquet(store_dir / RECORDS_NAME)
    assert len(out) == 2 and out["f_xy"].notna().sum() == 1


def test_apply_fills_every_copy_of_an_unlabelled_duplicate(tmp_path: Path) -> None:
    store_dir = tmp_path / "dup2"
    store_dir.mkdir(parents=True)
    frame = records_to_frame([_record("r1", PACKED_A), _record("r1", PACKED_A)])
    pq.write_table(frame_to_table(frame), store_dir / RECORDS_NAME)

    rep = apply(_csv(tmp_path, PACKED_A), store_dir)
    assert rep.n_populated == 1 and rep.wrote is True
    out = pd.read_parquet(store_dir / RECORDS_NAME)
    assert len(out) == 2                                   # no row lost or moved
    assert out["f_xy"].tolist() == pytest.approx([1.8247, 1.8247])
