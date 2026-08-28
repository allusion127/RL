"""``sklearn_fallback`` backend — vendor ``SurrogateEnsemble`` behind the
:class:`~lpopt.model.model_api.PositionValueModel` Protocol (plan sec. 4.5).

The primary campaign backend is :class:`PosValCnnBackend`.  This Extra-Trees
ensemble is its E2E safety net: it is used by the ``MODEL_HALT`` degradation
path (plan sec. 4.6 — two consecutive gate halts refit the fallback), and as a
lightweight, torch-free test double for the whole acquisition/campaign stack.

The vendor ensemble already returns a :class:`SurrogatePrediction` in the
``TARGET_NAMES`` 7-column layout, so ``predict`` is a thin adapter from the
Protocol's ``(patterns, case, cell)`` shape to the vendor's ``(patterns, cases,
cells)`` shape.

**Case one-hot is fit-frozen (plan sec. 4.5):** the vendor ``FeatureEncoder``
learns its case one-hot at ``fit`` time, so a ``predict`` for a case never seen
at fit raises ``KeyError``.  :meth:`fit_from_store` therefore pre-enumerates
*every* campaign case; :meth:`predict` additionally remaps any still-unknown
case to the fitted ``reference_case`` so the fallback can never abort the
campaign it exists to rescue.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.schema import unpack_pattern
from ..vendor.masterrl.domain import CaseKey, FOM, Pattern, PatternRecord
from ..vendor.masterrl.surrogate import SurrogateEnsemble, SurrogatePrediction

_PRIMARY = ("f_r", "f_q", "cbc_max", "cyclen")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(fv) else fv


def row_to_record(row: Any) -> PatternRecord | None:
    """Build a vendor :class:`PatternRecord` from one store row (or ``None``).

    Rows missing a usable primary FOM (``f_r`` / ``f_q`` / ``cbc_max`` /
    ``cyclen``) are dropped: the Extra-Trees ``_target_matrix`` reads those
    fields unconditionally, so a ``None`` would raise rather than be censored.
    Non-converged rows with finite last-iterate values are kept — the vendor
    ``fit`` censors them from every regression target and routes the pattern to
    the convergence classifier.
    """

    def _g(key: str) -> Any:
        getter = getattr(row, "get", None)
        return getter(key) if callable(getter) else row[key]

    primaries = {name: _finite(_g(name)) for name in _PRIMARY}
    if any(primaries[name] is None for name in _PRIMARY):
        return None
    try:
        pattern = unpack_pattern(str(_g("pattern")))
    except (ValueError, KeyError):
        return None
    pair = str(_g("case_pair"))
    feed = int(_g("feed"))
    cell = _finite(_g("e_core"))
    ao_abs = _finite(_g("ao_abs"))
    fom = FOM(
        f_r=primaries["f_r"],
        cbc_max=primaries["cbc_max"],
        f_q=primaries["f_q"],
        cyclen=primaries["cyclen"],
        ao_min=None if ao_abs is None else -ao_abs,
        ao_max=None if ao_abs is None else ao_abs,
        max_burnup=_finite(_g("max_assembly_burnup")),
        max_pin_burnup=_finite(_g("max_pin_burnup")),
        converged=bool(_g("converged")),
    )
    n_cycles = _finite(_g("n_cycles"))
    return PatternRecord(
        case=CaseKey(pair, feed),
        cell=float(cell) if cell is not None else 0.0,
        seed_id=str(_g("record_id"))[:16],
        pattern=pattern,
        fom=fom,
        ncyc=int(n_cycles) if n_cycles else 1,
        deck_path=Path("."),
        shf_path=Path("."),
    )


class SklearnBackend:
    """Extra-Trees ensemble adapter implementing ``PositionValueModel``."""

    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        *,
        library_id: str = "ga80",
        reference_case: CaseKey | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.library_id = library_id
        self.fitted_cases: set[CaseKey] = set(
            (ensemble.case_target_mean or {}).keys()
        )
        self.reference_case = reference_case or (
            next(iter(sorted(self.fitted_cases))) if self.fitted_cases else None
        )
        self._warned_remap = False

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def fit(
        cls,
        records: Sequence[PatternRecord],
        *,
        library_id: str = "ga80",
        n_estimators: int = 128,
        random_seed: int = 20260716,
    ) -> "SklearnBackend":
        if len(records) < 8:
            raise ValueError(
                f"sklearn_fallback needs >= 8 usable records, got {len(records)}"
            )
        ensemble = SurrogateEnsemble(
            n_estimators=n_estimators, random_seed=random_seed
        )
        ensemble.fit(list(records))
        reference = max(
            ((rec.case for rec in records)),
            key=lambda c: sum(1 for r in records if r.case == c),
        )
        return cls(ensemble, library_id=library_id, reference_case=reference)

    @classmethod
    def fit_from_store(
        cls,
        store_dir: str | Path,
        campaign_cases: Sequence[CaseKey],
        *,
        library_id: str = "ga80",
        extra_rows: Sequence[Any] = (),
        max_rows: int | None = None,
        **kwargs: Any,
    ) -> "SklearnBackend":
        """Fit from ``records.parquet`` rows for the campaign cases (+ extras).

        Every ``campaign_cases`` case that has store rows is pre-enumerated into
        the fit so its one-hot is frozen in.  ``extra_rows`` (e.g. this
        campaign's freshly verified labels) are appended verbatim.
        """

        from ..data.store import StoreReader

        wanted = set(campaign_cases)
        records: list[PatternRecord] = []
        reader = StoreReader(store_dir)
        try:
            df = reader.records
        except (FileNotFoundError, OSError):
            df = pd.DataFrame()
        if len(df):
            keys = list(zip(df["case_pair"].astype(str), df["feed"].astype(int)))
            mask = [CaseKey(p, f) in wanted for p, f in keys]
            sub = df[pd.Series(mask, index=df.index)]
            if max_rows is not None and len(sub) > max_rows:
                sub = sub.sample(n=max_rows, random_state=0)
            for _, row in sub.iterrows():
                rec = row_to_record(row)
                if rec is not None:
                    records.append(rec)
        for row in extra_rows:
            rec = row_to_record(row)
            if rec is not None:
                records.append(rec)
        if len(records) < 8:
            raise ValueError(
                "sklearn_fallback refit: fewer than 8 usable rows for campaign "
                f"cases {sorted(c.label for c in wanted)} "
                f"(found {len(records)}); cannot fit a fallback"
            )
        return cls.fit(records, library_id=library_id, **kwargs)

    # -- Protocol ----------------------------------------------------------- #
    def _map_cases(self, case: CaseKey, n: int) -> list[CaseKey]:
        if case in self.fitted_cases or not self.fitted_cases:
            return [case] * n
        if not self._warned_remap:
            self._warned_remap = True
            warnings.warn(
                f"sklearn_fallback: case {case.label} unseen at fit; remapping to "
                f"{self.reference_case.label if self.reference_case else '?'}",
                RuntimeWarning,
                stacklevel=2,
            )
        return [self.reference_case] * n  # type: ignore[list-item]

    def predict(
        self, patterns: Sequence[Pattern], case: CaseKey, cell: float = 0.0
    ) -> SurrogatePrediction:
        patterns = list(patterns)
        n = len(patterns)
        if n == 0:
            empty = np.zeros((0, 7))
            return SurrogatePrediction(empty, empty.copy(), empty.copy())
        cases = self._map_cases(case, n)
        cells = [float(cell)] * n
        return self.ensemble.predict(patterns, cases, cells)

    def predict_convergence(
        self, patterns: Sequence[Pattern], case: CaseKey, cell: float = 0.0
    ) -> np.ndarray:
        patterns = list(patterns)
        n = len(patterns)
        if n == 0:
            return np.zeros(0, dtype=float)
        cases = self._map_cases(case, n)
        cells = [float(cell)] * n
        return self.ensemble.predict_convergence_probability(patterns, cases, cells)

    def position_values(
        self, pattern: Pattern, case: CaseKey, cell: float = 0.0
    ) -> None:
        return None

    def finetune(
        self, new: Any, replay: Any, epochs: int = 3, seed: int = 0
    ) -> dict:
        """Refit the ensemble on ``new`` + ``replay`` + the fitted rows.

        Extra-Trees have no incremental update, so a "fine-tune" is a full
        refit that folds in the new labels while keeping the frozen case set
        (the previously fitted cases are re-supplied via the reference case if
        the fresh rows do not cover them).
        """

        new_rows = _as_rows(new)
        replay_rows = _as_rows(replay)
        records: list[PatternRecord] = []
        for row in [*new_rows, *replay_rows]:
            rec = row_to_record(row)
            if rec is not None:
                records.append(rec)
        n_seen = len(records)
        if n_seen < 8:
            return {"refit": False, "n_new": len(new_rows), "reason": "too few rows"}
        self.ensemble.fit(records)
        self.fitted_cases = set((self.ensemble.case_target_mean or {}).keys())
        return {"refit": True, "n_rows": n_seen, "n_members": 1}

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.ensemble.save(out / "ensemble.joblib")
        return out

    @classmethod
    def load(cls, path: str | Path, *, library_id: str = "ga80") -> "SklearnBackend":
        ensemble = SurrogateEnsemble.load(Path(path) / "ensemble.joblib")
        return cls(ensemble, library_id=library_id)


def _as_rows(rows: Any) -> list[Any]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return [r for _, r in rows.iterrows()]
    return list(rows)


__all__ = ["SklearnBackend", "row_to_record"]
