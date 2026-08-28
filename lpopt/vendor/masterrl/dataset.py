"""Loader and validation for ``FEASIBLE_PACKAGE`` warm seeds."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

from .domain import CaseKey, FOM, Pattern, PatternRecord


MANIFEST_COLUMNS = (
    "id",
    "pair",
    "feed",
    "cell",
    "F_r",
    "CBC_max",
    "F_q",
    "cyclen",
    "ncyc",
)

# Optional manifest columns: absent in the original hand-packaged manifest but
# written by ``master_rl.ingest`` when the GA archive provides the values.
OPTIONAL_MANIFEST_COLUMNS = (
    "AO_min",
    "AO_max",
    "max_assembly_burnup",
    "max_pin_burnup",
    "eq_ok",
)


def _merge_physical_records(
    *groups: Sequence[PatternRecord],
) -> list[PatternRecord]:
    """Canonical-keyed upsert of physical labels; later groups win.

    One physical core must carry exactly one label per case: a verified
    baseline (or resume replay) record REPLACES the manifest row of the same
    ``(case, canonical)`` key instead of duplicating it, so surrogate
    training and incumbent selection always see the most-verified FOM (F-03,
    N-8).
    """

    merged: dict[tuple[CaseKey, str], PatternRecord] = {}
    for group in groups:
        for record in group:
            merged[(record.case, record.pattern.canonical())] = record
    return list(merged.values())


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return float(raw)


def _optional_bool(row: Mapping[str, str], key: str, default: bool = True) -> bool:
    """Parse an optional ``"True"``/``"False"``/empty manifest cell."""

    raw = row.get(key)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    raise ValueError(f"manifest column {key!r} must be True/False, got {raw!r}")


@dataclass(frozen=True, slots=True)
class CaseData:
    key: CaseKey
    cell: float
    records: tuple[PatternRecord, ...]
    template_path: Path
    restart_path: Path

    @property
    def seed_count(self) -> int:
        return len(self.records)

    @property
    def best_cycle_seed(self) -> PatternRecord:
        # Prefer converged (eq_ok) seeds: a non-converged manifest row carries
        # last-iterate values, not physics, and must not become the incumbent
        # while a converged alternative exists.
        converged = [record for record in self.records if record.fom.converged]
        pool = converged or self.records
        return max(pool, key=lambda record: record.fom.cyclen)

    def incumbent_seed(self, constraints=None) -> PatternRecord:
        """Mode-aware incumbent (W3): duck-typed to avoid a reward import.

        ``trade_off`` (or no constraints) keeps ``best_cycle_seed``; target
        mode returns the converged seed with the smallest |cyclen − target|.
        """

        if constraints is None or (
            getattr(constraints, "objective_mode", "trade_off") != "target_cycle"
        ):
            return self.best_cycle_seed
        target = float(constraints.cycle_target_efpd)
        converged = [record for record in self.records if record.fom.converged]
        pool = converged or self.records
        return min(pool, key=lambda record: abs(record.fom.cyclen - target))

    def top_seeds(self, count: int = 5) -> tuple[PatternRecord, ...]:
        return tuple(
            sorted(self.records, key=lambda record: record.fom.cyclen, reverse=True)[:count]
        )


class PackageDataset:
    """Validated in-memory view of all GA-Surrogate warm seeds."""

    def __init__(self, root: Path, records: Sequence[PatternRecord]) -> None:
        self.root = root.resolve()
        self.records = tuple(records)
        grouped: dict[CaseKey, list[PatternRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.case].append(record)
        cases: dict[CaseKey, CaseData] = {}
        for key, case_records in sorted(grouped.items()):
            cells = {record.cell for record in case_records}
            if len(cells) != 1:
                raise ValueError(f"case {key.label} has inconsistent cell values: {cells}")
            templates = {record.deck_path for record in case_records}
            base_dir = self.root / "bases" / key.folder
            restarts = tuple(base_dir.glob("MAS_RST.*"))
            if len(restarts) != 1:
                raise ValueError(
                    f"case {key.label} requires exactly one base restart; found {len(restarts)}"
                )
            # Any seed deck is a valid template; only LPD_SHF and descriptive metadata vary.
            template = sorted(templates)[0]
            cases[key] = CaseData(
                key=key,
                cell=next(iter(cells)),
                records=tuple(case_records),
                template_path=template,
                restart_path=restarts[0],
            )
        self.cases: Mapping[CaseKey, CaseData] = cases
        self._lookup = {
            (record.case, record.pattern.canonical()): record for record in self.records
        }

    @classmethod
    def load(cls, root: str | Path, *, strict: bool = True) -> "PackageDataset":
        root_path = Path(root).resolve()
        manifest = root_path / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"manifest is missing columns: {sorted(missing)}")
            manifest_rows = list(reader)

        records: list[PatternRecord] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(manifest_rows, start=2):
            seed_id = row["id"].strip()
            if seed_id in seen_ids:
                raise ValueError(f"duplicate seed id {seed_id!r} at manifest line {line_number}")
            seen_ids.add(seed_id)
            try:
                key = CaseKey(row["pair"].strip(), int(row["feed"]))
                cell = float(row["cell"])
                fom = FOM(
                    f_r=float(row["F_r"]),
                    cbc_max=float(row["CBC_max"]),
                    f_q=float(row["F_q"]),
                    cyclen=float(row["cyclen"]),
                    ao_min=_optional_float(row, "AO_min"),
                    ao_max=_optional_float(row, "AO_max"),
                    max_burnup=_optional_float(row, "max_assembly_burnup"),
                    max_pin_burnup=_optional_float(row, "max_pin_burnup"),
                    converged=_optional_bool(row, "eq_ok"),
                )
                ncyc = int(row["ncyc"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value at manifest line {line_number}") from error

            seed_dir = root_path / "cores" / key.folder / seed_id
            shf_path = seed_dir / "loading_shf.txt"
            decks = tuple(seed_dir.glob("MAS_INP_cy*.inp"))
            if not shf_path.is_file() or len(decks) != 1:
                raise FileNotFoundError(
                    f"seed {seed_id}: expected loading_shf.txt and one MAS_INP_cy*.inp"
                )
            pattern = Pattern.from_file(shf_path)
            if strict:
                pattern.validate_case(key.pair, key.feed)
                pattern.validate_quarter_conventions()
            records.append(
                PatternRecord(
                    case=key,
                    cell=cell,
                    seed_id=seed_id,
                    pattern=pattern,
                    fom=fom,
                    ncyc=ncyc,
                    deck_path=decks[0],
                    shf_path=shf_path,
                )
            )

        dataset = cls(root_path, records)
        if strict:
            core_dirs = {
                path.parent.resolve()
                for path in (root_path / "cores").glob("*/*/loading_shf.txt")
            }
            record_dirs = {record.shf_path.parent.resolve() for record in records}
            if core_dirs != record_dirs:
                missing = sorted(str(path) for path in core_dirs - record_dirs)
                extra = sorted(str(path) for path in record_dirs - core_dirs)
                raise ValueError(
                    f"manifest/core mismatch: unlisted={missing[:3]}, missing={extra[:3]}"
                )
        return dataset

    def case(self, pair: str, feed: int) -> CaseData:
        key = CaseKey(pair, int(feed))
        try:
            return self.cases[key]
        except KeyError as error:
            available = ", ".join(case.label for case in self.cases)
            raise KeyError(f"unknown case {key.label}; available: {available}") from error

    def find_known(self, case: CaseKey, pattern: Pattern) -> PatternRecord | None:
        return self._lookup.get((case, pattern.canonical()))

    def summary(self) -> dict[str, object]:
        metric_names = ("f_r", "cbc_max", "f_q", "cyclen")
        metrics: dict[str, dict[str, float]] = {}
        for name in metric_names:
            values = [float(getattr(record.fom, name)) for record in self.records]
            metrics[name] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }

        cases: list[dict[str, object]] = []
        for key, data in self.cases.items():
            best = data.best_cycle_seed
            cases.append(
                {
                    "pair": key.pair,
                    "feed": key.feed,
                    "cell": data.cell,
                    "seeds": data.seed_count,
                    "best_seed_id": best.seed_id,
                    "best_seed_fom": best.fom.as_dict(),
                    "fresh_card_counts": sorted(
                        {record.pattern.fresh_card_count for record in data.records}
                    ),
                    "weighted_feeds": sorted({record.pattern.feed for record in data.records}),
                }
            )
        strata = Counter((record.cell, record.case.feed) for record in self.records)
        return {
            "root": str(self.root),
            "records": len(self.records),
            "case_count": len(self.cases),
            "strata": [
                {"cell": cell, "feed": feed, "seeds": count}
                for (cell, feed), count in sorted(strata.items())
            ],
            "metrics": metrics,
            "cases": cases,
        }

    def write_summary(self, path: str | Path) -> Path:
        from .jsonio import dumps_strict

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dumps_strict(self.summary()), encoding="utf-8")
        return target
