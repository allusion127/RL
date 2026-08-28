"""Split-manifest family (plan sec. 4.4): random / ancestry-group / leave-pair /
feed-filter / e_core-band splits are deterministic under seed, never straddle a
group across S1 train/val, keep the S2 holdout pairs out of train, and reload
byte-identically from JSON."""

from __future__ import annotations

from pathlib import Path

import pytest

from lpopt.model.splits import (
    SplitManifest,
    make_e_core_band,
    make_feed_filter,
    make_leave_pair,
    make_s0,
    make_s1,
    make_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS = REPO_ROOT / "data" / "store" / "records.parquet"


def _df():
    if not RECORDS.is_file():
        pytest.skip("Dataset-A store not present")
    import pandas as pd

    return pd.read_parquet(RECORDS)


@pytest.fixture(scope="module")
def df():
    return _df()


# --------------------------------------------------------------------------- #
# family driver
# --------------------------------------------------------------------------- #
def test_make_splits_emits_full_family(df, tmp_path) -> None:
    manifests = make_splits(df, seed=13, out_dir=tmp_path)
    assert set(manifests) == {"S0", "S1", "S2", "S3a", "S3b", "S4"}
    for name in manifests:
        assert (tmp_path / f"{name}.json").is_file()
    # every train/val pair is a disjoint partition of a subset of the store
    ids = set(df["record_id"].astype(str))
    for m in manifests.values():
        assert set(m.train_ids).isdisjoint(set(m.val_ids))
        assert set(m.train_ids) | set(m.val_ids) <= ids


# --------------------------------------------------------------------------- #
# S0 random 90/10 + determinism
# --------------------------------------------------------------------------- #
def test_s0_random_90_10_and_deterministic(df) -> None:
    a = make_s0(df, seed=42)
    b = make_s0(df, seed=42)
    c = make_s0(df, seed=43)
    total = a.n_train + a.n_val
    assert total == len(df)
    assert a.n_val == pytest.approx(round(0.10 * total), abs=1)
    assert a.to_dict() == b.to_dict()          # deterministic under a fixed seed
    assert a.val_ids != c.val_ids              # a different seed reshuffles


# --------------------------------------------------------------------------- #
# S1 ancestry-closure group split — no group straddles train/val
# --------------------------------------------------------------------------- #
def test_s1_no_group_straddles_train_val(df) -> None:
    m = make_s1(df, seed=7)
    train = set(m.train_ids)
    val = set(m.val_ids)
    assert train and val
    assert train.isdisjoint(val)

    # rebuild the union-find groups and assert each lands entirely on one side
    from lpopt.model.splits import _UnionFind
    import pandas as pd

    ids = df["record_id"].astype(str).tolist()
    uf = _UnionFind(ids)
    by_campaign: dict[str, list[str]] = {}
    for rid, camp in zip(df["record_id"].astype(str), df["campaign"]):
        by_campaign.setdefault("c::" + str(camp), []).append(rid)
    for members in by_campaign.values():
        for other in members[1:]:
            uf.union(members[0], other)
    id_set = set(ids)
    for rid, parent in zip(df["record_id"].astype(str), df["parent_record_id"]):
        if parent is not None and not pd.isna(parent) and str(parent) in id_set:
            uf.union(rid, str(parent))

    for members in uf.groups().values():
        in_val = [mid in val for mid in members]
        assert all(in_val) or not any(in_val), "a group straddled train/val"


def test_s1_deterministic(df) -> None:
    assert make_s1(df, seed=99).to_dict() == make_s1(df, seed=99).to_dict()


# --------------------------------------------------------------------------- #
# S2 leave-pair-out — holdout pairs entirely absent from train
# --------------------------------------------------------------------------- #
def test_s2_holdout_pairs_absent_from_train(df) -> None:
    pairs = ("C3_C6", "A01_B05")
    m = make_leave_pair(df, seed=0, holdout_pairs=pairs)
    train = df[df["record_id"].astype(str).isin(set(m.train_ids))]
    assert not train["case_pair"].astype(str).isin(pairs).any()
    # and every record of those pairs is in val
    n_pairs = int(df["case_pair"].astype(str).isin(pairs).sum())
    assert m.n_val == n_pairs
    assert n_pairs > 0


def test_s2_parameterized_extra_pair(df) -> None:
    # an absent Dataset-B pair is tolerated (contributes zero val rows)
    m = make_leave_pair(df, seed=0, holdout_pairs=("C3_C6", "ZZ9_ZZ8"))
    assert m.n_val == int(df["case_pair"].astype(str).eq("C3_C6").sum())


# --------------------------------------------------------------------------- #
# S3a / S3b / S4 filter handles (emitted even when empty)
# --------------------------------------------------------------------------- #
def test_s3a_feed117_filter(df) -> None:
    m = make_feed_filter(df, seed=0, name="S3a", feeds=[117])
    n_117 = int(df["feed"].astype(int).eq(117).sum())
    assert m.n_val == n_117
    assert m.status == ("ok" if n_117 else "empty")
    assert m.n_train == len(df) - n_117        # still a usable handle when empty


def test_s3b_awaiting_production(df) -> None:
    m = make_feed_filter(df, seed=0, name="S3b", feeds=[105, 113],
                         status_when_empty="awaiting_production")
    n_prod = int(df["feed"].astype(int).isin({105, 113}).sum())
    assert m.n_val == n_prod
    # empty on the current feed-121 store -> flagged for production
    assert m.status == ("ok" if n_prod else "awaiting_production")
    assert m.predicate["feed_in"] == [105, 113]


def test_s4_e_core_band_holdout(df) -> None:
    m = make_e_core_band(df, seed=0, lo=5.43, hi=5.50)
    import pandas as pd

    e = pd.to_numeric(df["e_core"], errors="coerce")
    expected = int(((e >= 5.43) & (e < 5.50)).sum())
    assert m.n_val == expected
    assert m.predicate["e_core_band"] == [5.43, 5.50]


# --------------------------------------------------------------------------- #
# JSON round-trip
# --------------------------------------------------------------------------- #
def test_manifests_reload_identically(df, tmp_path) -> None:
    manifests = make_splits(df, seed=5, out_dir=tmp_path)
    for name, m in manifests.items():
        reloaded = SplitManifest.from_json(tmp_path / f"{name}.json")
        assert reloaded.to_dict() == m.to_dict()
        assert reloaded.record_ids("train") == m.train_ids
        assert reloaded.record_ids("val") == m.val_ids
