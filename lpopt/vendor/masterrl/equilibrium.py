"""Successive-cycle equilibrium reconvergence for one loading-pattern candidate.

``FEASIBLE_PACKAGE`` says its original evaluator required two consecutive
five-metric comparisons, but the referenced ``master/ga_eval.py`` (and hence
its exact tolerances) is not included.  This module therefore makes the five
absolute tolerances explicit and configurable.  The provisional defaults are
defined by :class:`EquilibriumTolerances`; production reproduction should pass
the original values if they become available.

Each candidate evaluation is distinct from a raw MASTER call: one candidate
normally requires several chained cycle calculations.  Both counts are kept
separately by :class:`EquilibriumRunner` and reported in its result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
from typing import Mapping

from .dataset import CaseData
from .domain import Pattern
from .burnup import enable_ppi_output
from .master import (
    DeckFormatError,
    MasterMetrics,
    MasterRunError,
    MasterRunner,
    extract_lpd_shf,
    replace_lpd_shf,
)


TOLERANCE_KEYS: tuple[str, ...] = ("cyclen", "cbc_max", "f_q", "f_r", "ao")


@dataclass(frozen=True, slots=True)
class EquilibriumTolerances:
    """Absolute successive-cycle tolerances for the five available FOMs.

    ``ao`` is the larger of the ``AO_min`` and ``AO_max`` endpoint changes.
    A value of ``None`` disables that metric.  Defaults are deliberately
    visible and provisional because the package omits ``ga_eval.py``:

    * cycle length: 0.10 EFPD
    * maximum CBC: 1.0 ppm
    * Fq, Fr, and AO envelope: 1e-3 each
    """

    cyclen: float | None = 0.10
    cbc_max: float | None = 1.0
    f_q: float | None = 1.0e-3
    f_r: float | None = 1.0e-3
    ao: float | None = 1.0e-3

    def __post_init__(self) -> None:
        active = 0
        for name in TOLERANCE_KEYS:
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} tolerance must be finite and non-negative")
            active += 1
        if active == 0:
            raise ValueError("at least one equilibrium tolerance must be enabled")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, float | None]
    ) -> "EquilibriumTolerances":
        """Build tolerances with exactly the supplied metric keys enabled."""

        unknown = set(values) - set(TOLERANCE_KEYS)
        if unknown:
            raise ValueError(f"unknown equilibrium tolerance keys: {sorted(unknown)}")
        configured = {name: values.get(name) for name in TOLERANCE_KEYS}
        return cls(**configured)

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in TOLERANCE_KEYS}


DEFAULT_EQUILIBRIUM_TOLERANCES = EquilibriumTolerances()


@dataclass(frozen=True, slots=True)
class EquilibriumComparison:
    previous_cycle: int
    current_cycle: int
    deltas: Mapping[str, float]
    within_tolerance: bool


@dataclass(frozen=True, slots=True)
class EquilibriumCycle:
    cycle: int
    metrics: MasterMetrics
    restart_path: Path
    work_dir: Path


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    """One candidate's complete reconvergence history and cost accounting."""

    cycles: tuple[EquilibriumCycle, ...]
    comparisons: tuple[EquilibriumComparison, ...]
    converged: bool
    consecutive_matches: int
    required_consecutive: int
    tolerances: EquilibriumTolerances
    candidate_evaluations: int
    master_process_calls: int
    retained_work_dirs: tuple[Path, ...]
    # Cycle cap the runner was configured with (F-10).  Defaulted for legacy
    # payloads/pickles that predate cap-hit reporting.
    max_cycles: int = 0

    @property
    def metrics(self) -> MasterMetrics:
        return self.cycles[-1].metrics

    @property
    def fom(self):
        """Final metrics converted to the reward-domain FOM."""

        return self.metrics.to_fom()

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def converged_at_cap(self) -> bool:
        """Converged, but only on the very last allowed cycle (F-10).

        A cap-exact convergence is not the same evidence as a comfortable one:
        it means the chain had zero slack, so the configured ``max_cycles``
        should be reviewed even though the result formally converged.
        """

        return self.converged and self.n_cycles == self.max_cycles

    @property
    def cap_exhausted(self) -> bool:
        """The cycle budget ran out before consecutive comparisons settled."""

        return not self.converged

    @property
    def tolerance_margin(self) -> float | None:
        """Worst last-comparison delta/tolerance ratio over the active axes.

        ``0.02`` means the tightest metric still had 50x headroom; values
        approaching 1.0 flag a marginal convergence.  ``None`` when no
        comparison was made (single-cycle run).
        """

        if not self.comparisons:
            return None
        ratios: list[float] = []
        for name, delta in self.comparisons[-1].deltas.items():
            tolerance = getattr(self.tolerances, name)
            if tolerance is None:
                continue
            if tolerance > 0.0:
                ratios.append(delta / tolerance)
            else:
                ratios.append(float("inf") if delta > 0.0 else 0.0)
        return max(ratios) if ratios else None


class EquilibriumRunError(MasterRunError):
    """A chaining/deck failure whose relevant work directory is retained."""


_CARD_HEADER = re.compile(r"^[ \t]*%(?P<name>[^\s\r\n]+)", re.IGNORECASE)
_TOKEN_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<token>\S+)(?P<tail>[ \t]*(?:#.*)?)(?P<eol>\r\n|\n|\r)?$"
)
_JOB_IDE_LINE = re.compile(
    r"^(?P<prefix>[ \t]*\S+[ \t]+)(?P<cycle>[+-]?\d+)"
    r"(?P<tail>[^\r\n]*)(?P<eol>\r\n|\n|\r)?$"
)
_CY_TOKEN = re.compile(r"\bCY(?P<number>\d+)\b", re.IGNORECASE)


def _card_span(lines: list[str], card: str) -> tuple[int, int]:
    matches: list[int] = []
    for index, line in enumerate(lines):
        match = _CARD_HEADER.match(line)
        if match is not None and match.group("name").upper() == card.upper():
            matches.append(index)
    if len(matches) != 1:
        raise DeckFormatError(
            f"deck must contain exactly one %{card} card; found {len(matches)}"
        )
    start = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _CARD_HEADER.match(lines[index]) is not None:
            end = index
            break
    return start, end


def _data_line_indices(lines: list[str], start: int, end: int) -> list[int]:
    indices: list[int] = []
    for index in range(start + 1, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or stripped == "/":
            continue
        indices.append(index)
    return indices


def deck_cycle(deck: str) -> int:
    """Return the integer cycle in the sole ``%JOB_IDE`` card."""

    lines = deck.splitlines(keepends=True)
    start, end = _card_span(lines, "JOB_IDE")
    data = _data_line_indices(lines, start, end)
    if not data:
        raise DeckFormatError("%JOB_IDE has no data line")
    match = _JOB_IDE_LINE.match(lines[data[0]])
    if match is None:
        raise DeckFormatError("could not parse plant/cycle fields in %JOB_IDE")
    cycle = int(match.group("cycle"))
    if cycle < 0:
        raise DeckFormatError("%JOB_IDE cycle must be non-negative")
    return cycle


def _safe_restart_name(restart_name: str) -> str:
    if (
        not restart_name
        or Path(restart_name).name != restart_name
        or any(character in restart_name for character in ("/", "\\", "#", "%"))
        or any(character.isspace() for character in restart_name)
        or not restart_name.upper().startswith("MAS_RST.")
    ):
        raise ValueError("restart_name must be a safe MAS_RST.* basename")
    return restart_name


def advance_cycle_deck(deck: str, restart_name: str, cycle: int) -> str:
    """Update only the single restart reference, cycle, and safe CY title.

    The packaged equilibrium decks use exactly one restart.  Refusing a more
    complex ``%JOB_TYP`` avoids silently changing the wrong restart in a
    multi-restart deck.  A descriptive title is changed only when it contains
    exactly one ``CYnn`` token matching the old ``%JOB_IDE`` cycle.
    """

    restart_name = _safe_restart_name(restart_name)
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
        raise ValueError("cycle must be a non-negative integer")

    lines = deck.splitlines(keepends=True)
    old_cycle = deck_cycle(deck)

    typ_start, typ_end = _card_span(lines, "JOB_TYP")
    typ_data = _data_line_indices(lines, typ_start, typ_end)
    if not typ_data:
        raise DeckFormatError("%JOB_TYP has no data line")
    typ_fields = lines[typ_data[0]].split("#", 1)[0].split()
    try:
        restart_count = int(typ_fields[0])
    except (IndexError, ValueError) as exc:
        raise DeckFormatError("could not parse restart count in %JOB_TYP") from exc
    if restart_count != 1 or len(typ_data) < 2:
        raise DeckFormatError(
            "equilibrium chaining requires exactly one %JOB_TYP restart"
        )
    restart_index = typ_data[1]
    restart_match = _TOKEN_LINE.match(lines[restart_index])
    if restart_match is None:
        raise DeckFormatError("could not parse restart reference in %JOB_TYP")
    lines[restart_index] = (
        restart_match.group("indent")
        + restart_name
        + restart_match.group("tail")
        + (restart_match.group("eol") or "")
    )

    ide_start, ide_end = _card_span(lines, "JOB_IDE")
    ide_data = _data_line_indices(lines, ide_start, ide_end)
    if not ide_data:
        raise DeckFormatError("%JOB_IDE has no data line")
    ide_index = ide_data[0]
    ide_match = _JOB_IDE_LINE.match(lines[ide_index])
    if ide_match is None:
        raise DeckFormatError("could not parse plant/cycle fields in %JOB_IDE")
    lines[ide_index] = (
        ide_match.group("prefix")
        + str(cycle)
        + ide_match.group("tail")
        + (ide_match.group("eol") or "")
    )

    # Updating a title is descriptive only.  Ambiguous/mismatched text remains
    # byte-for-byte unchanged rather than risking a semantic deck edit.
    try:
        title_start, title_end = _card_span(lines, "JOB_TIT")
    except DeckFormatError:
        title_start = title_end = -1
    if title_start >= 0:
        title_data = _data_line_indices(lines, title_start, title_end)
        title_matches = [
            (index, match)
            for index in title_data
            for match in _CY_TOKEN.finditer(lines[index])
        ]
        if len(title_matches) == 1:
            title_index, title_match = title_matches[0]
            if int(title_match.group("number")) == old_cycle:
                old_number = title_match.group("number")
                replacement = str(cycle).zfill(len(old_number))
                title_line = lines[title_index]
                lines[title_index] = (
                    title_line[: title_match.start("number")]
                    + replacement
                    + title_line[title_match.end("number") :]
                )

    return "".join(lines)


def _metric_deltas(
    previous: MasterMetrics,
    current: MasterMetrics,
    tolerances: EquilibriumTolerances,
) -> dict[str, float]:
    deltas = {
        "cyclen": abs(current.cyclen - previous.cyclen),
        "cbc_max": abs(current.cbc_max - previous.cbc_max),
        "f_q": abs(current.f_q - previous.f_q),
        "f_r": abs(current.f_r - previous.f_r),
        "ao": max(
            abs(current.ao_min - previous.ao_min),
            abs(current.ao_max - previous.ao_max),
        ),
    }
    return {
        name: value
        for name, value in deltas.items()
        if getattr(tolerances, name) is not None
    }


class EquilibriumRunner:
    """Chain MASTER cycles until successive five-FOM comparisons settle."""

    def __init__(
        self,
        master_runner: MasterRunner,
        *,
        max_cycles: int = 10,
        consecutive: int = 2,
        tolerances: EquilibriumTolerances
        | Mapping[str, float | None]
        | None = None,
        keep_success: bool = False,
        enable_pin_burnup: bool = False,
    ) -> None:
        if not isinstance(max_cycles, int) or isinstance(max_cycles, bool) or max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 1:
            raise ValueError("consecutive must be a positive integer")
        if tolerances is None:
            resolved_tolerances = DEFAULT_EQUILIBRIUM_TOLERANCES
        elif isinstance(tolerances, EquilibriumTolerances):
            resolved_tolerances = tolerances
        else:
            resolved_tolerances = EquilibriumTolerances.from_mapping(tolerances)

        self.master_runner = master_runner
        self.max_cycles = max_cycles
        self.consecutive = consecutive
        self.tolerances = resolved_tolerances
        self.keep_success = bool(keep_success)
        self.enable_pin_burnup = bool(enable_pin_burnup)
        self.candidate_evaluations = 0
        self.master_process_calls = 0

    @property
    def evaluation_count(self) -> int:
        return self.candidate_evaluations

    @property
    def raw_process_calls(self) -> int:
        return self.master_process_calls

    @staticmethod
    def _read_deck(path: Path) -> str:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "cp949", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise AssertionError("latin-1 decoding is total")

    @staticmethod
    def _generated_restart(work_dir: Path, input_restart: Path) -> Path:
        generated = sorted(
            path
            for path in work_dir.glob("MAS_RST.*")
            if path.is_file() and path.name != input_restart.name
        )
        if len(generated) != 1:
            raise EquilibriumRunError(
                "expected exactly one newly generated MAS_RST.* file, "
                f"found {len(generated)}",
                work_dir=work_dir,
            )
        return generated[0]

    @staticmethod
    def _clean(paths: list[Path]) -> None:
        for path in paths:
            try:
                shutil.rmtree(path)
            except OSError:
                # A retained success is preferable to hiding valid metrics
                # behind a cleanup-only failure.
                pass

    def run(self, case_data: CaseData, pattern: Pattern) -> EquilibriumResult:
        """Evaluate one candidate from the case template and base restart.

        MASTER's metric cache is intentionally bypassed: a cached scalar result
        has no restart artifact with which to begin the next cycle.  Successful
        work directories are retained while chaining, then removed unless this
        runner was constructed with ``keep_success=True``.  MASTER execution
        failures and restart-discovery failures always retain their own work
        directory for diagnosis.
        """

        self.candidate_evaluations += 1
        raw_before = self.master_runner.raw_process_calls
        successful_dirs: list[Path] = []
        cycles: list[EquilibriumCycle] = []
        comparisons: list[EquilibriumComparison] = []
        consecutive_matches = 0
        converged = False

        try:
            pattern.validate_case(case_data.key.pair, case_data.key.feed)
            pattern.validate_quarter_conventions()
            deck = self._read_deck(case_data.template_path)
            deck = replace_lpd_shf(deck, pattern.to_shf())
            expected_shf = extract_lpd_shf(deck)
            cycle_number = deck_cycle(deck)
            restart = case_data.restart_path.expanduser().resolve()

            for _ in range(self.max_cycles):
                if (
                    self.enable_pin_burnup
                    and consecutive_matches >= max(0, self.consecutive - 1)
                ):
                    deck = enable_ppi_output(deck)
                result = self.master_runner.run(
                    case_data.template_path.parent,
                    deck_text=deck,
                    restart_path=restart,
                    use_cache=False,
                    keep_success=True,
                )
                if result.work_dir is None:
                    raise AssertionError("uncached retained MASTER run has no work_dir")
                work_dir = result.work_dir
                generated_restart = self._generated_restart(work_dir, restart)
                successful_dirs.append(work_dir)
                cycles.append(
                    EquilibriumCycle(
                        cycle=cycle_number,
                        metrics=result.metrics,
                        restart_path=generated_restart,
                        work_dir=work_dir,
                    )
                )

                if len(cycles) > 1:
                    previous = cycles[-2]
                    current = cycles[-1]
                    deltas = _metric_deltas(
                        previous.metrics, current.metrics, self.tolerances
                    )
                    within = all(
                        delta <= getattr(self.tolerances, name)
                        for name, delta in deltas.items()
                    )
                    comparisons.append(
                        EquilibriumComparison(
                            previous_cycle=previous.cycle,
                            current_cycle=current.cycle,
                            deltas=deltas,
                            within_tolerance=within,
                        )
                    )
                    consecutive_matches = consecutive_matches + 1 if within else 0
                    if consecutive_matches >= self.consecutive:
                        converged = True
                        break

                if len(cycles) < self.max_cycles:
                    cycle_number += 1
                    deck = advance_cycle_deck(
                        deck, generated_restart.name, cycle_number
                    )
                    if extract_lpd_shf(deck) != expected_shf:
                        raise AssertionError("cycle advancement changed %LPD_SHF")
                    restart = generated_restart

            calls = self.master_runner.raw_process_calls - raw_before
            if not self.keep_success:
                self._clean(successful_dirs)
            retained = tuple(path for path in successful_dirs if path.is_dir())
            return EquilibriumResult(
                cycles=tuple(cycles),
                comparisons=tuple(comparisons),
                converged=converged,
                consecutive_matches=consecutive_matches,
                required_consecutive=self.consecutive,
                tolerances=self.tolerances,
                candidate_evaluations=1,
                master_process_calls=calls,
                retained_work_dirs=retained,
                max_cycles=self.max_cycles,
            )
        except Exception:
            if not self.keep_success:
                self._clean(successful_dirs)
            raise
        finally:
            self.master_process_calls += (
                self.master_runner.raw_process_calls - raw_before
            )

    evaluate = run


__all__ = [
    "DEFAULT_EQUILIBRIUM_TOLERANCES",
    "EquilibriumComparison",
    "EquilibriumCycle",
    "EquilibriumResult",
    "EquilibriumRunError",
    "EquilibriumRunner",
    "EquilibriumTolerances",
    "TOLERANCE_KEYS",
    "advance_cycle_deck",
    "deck_cycle",
]
