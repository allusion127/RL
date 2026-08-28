"""Strict MASTER deck integration, summary parsing, and optional execution.

The reinforcement-learning code should see MASTER as a deterministic mapping
from a loading pattern to figures of merit.  This module keeps that boundary
small and independently testable:

* :func:`extract_lpd_shf` and :func:`replace_lpd_shf` operate on deck text and
  preserve every character outside the ``%LPD_SHF`` body;
* :func:`parse_mas_sum` reads SUMMARY EDIT 2/3 by their column headers; and
* :class:`MasterRunner` optionally stages and runs a user-supplied executable
  in an isolated directory with a content-addressed JSON cache.

No MASTER executable path is assumed or discovered at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # Avoid making this integration layer depend on the domain layer.
    from .domain import FOM


PathLike = str | os.PathLike[str]
Command = PathLike | Sequence[PathLike]

DEFAULT_CONVERGENCE_LIMIT = 1.0e-5
_CACHE_SCHEMA = 2
_HASH_CHUNK_SIZE = 1024 * 1024


class MasterError(Exception):
    """Base class for deterministic MASTER integration failures."""


class DeckFormatError(MasterError, ValueError):
    """Raised when a deck or replacement shuffle block is malformed."""


class SummaryParseError(MasterError, ValueError):
    """Raised when ``MAS_SUM`` is absent, malformed, or unconverged."""


class MasterRunError(MasterError, RuntimeError):
    """A staging or execution failure with the retained work directory.

    ``work_dir`` is ``None`` for pre-flight failures.  Once a case has been
    staged, it is deliberately retained on every failure for diagnosis.
    """

    def __init__(self, message: str, *, work_dir: Path | None = None) -> None:
        if work_dir is not None:
            message = f"{message} (work directory retained at {work_dir})"
        super().__init__(message)
        self.work_dir = work_dir


@dataclass(frozen=True, slots=True)
class MasterMetrics:
    """Validated figures of merit parsed from MASTER SUMMARY EDIT 2/3.

    Attributes use the naming already used by the RL domain model.  ``cyclen``
    is the last (and, after monotonicity validation, maximum) unique EFPD.
    ``errflx_max`` is the maximum absolute EDIT 2 flux residual.  The parser
    only constructs this dataclass after all values are finite and the maximum
    residual does not exceed the configured convergence limit.
    """

    cyclen: float
    cbc_max: float
    f_q: float
    f_r: float
    ao_min: float
    ao_max: float
    errflx_max: float
    n_steps: int
    max_assembly_burnup: float | None = None
    max_pin_burnup: float | None = None
    max_burnup_assembly: str | None = None
    max_burnup_pin: str | None = None

    @property
    def cycle_length(self) -> float:
        """Descriptive alias for :attr:`cyclen`."""

        return self.cyclen

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Return a stable, JSON-compatible representation."""

        return {
            "cyclen": self.cyclen,
            "cbc_max": self.cbc_max,
            "f_q": self.f_q,
            "f_r": self.f_r,
            "ao_min": self.ao_min,
            "ao_max": self.ao_max,
            "errflx_max": self.errflx_max,
            "n_steps": self.n_steps,
            "max_assembly_burnup": self.max_assembly_burnup,
            "max_pin_burnup": self.max_pin_burnup,
            "max_burnup_assembly": self.max_burnup_assembly,
            "max_burnup_pin": self.max_burnup_pin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MasterMetrics":
        """Reconstruct and validate cached metrics."""

        try:
            metrics = cls(
                cyclen=float(data["cyclen"]),
                cbc_max=float(data["cbc_max"]),
                f_q=float(data["f_q"]),
                f_r=float(data["f_r"]),
                ao_min=float(data["ao_min"]),
                ao_max=float(data["ao_max"]),
                errflx_max=float(data["errflx_max"]),
                n_steps=int(data["n_steps"]),
                max_assembly_burnup=(
                    None
                    if data.get("max_assembly_burnup") is None
                    else float(data["max_assembly_burnup"])
                ),
                max_pin_burnup=(
                    None
                    if data.get("max_pin_burnup") is None
                    else float(data["max_pin_burnup"])
                ),
                max_burnup_assembly=(
                    None
                    if data.get("max_burnup_assembly") is None
                    else str(data["max_burnup_assembly"])
                ),
                max_burnup_pin=(
                    None
                    if data.get("max_burnup_pin") is None
                    else str(data["max_burnup_pin"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SummaryParseError("cached MASTER metrics are malformed") from exc
        _validate_metrics(metrics)
        return metrics

    def to_fom(self) -> "FOM":
        """Convert to the package's reward-domain :class:`~master_rl.domain.FOM`."""

        from .domain import FOM

        return FOM(
            f_r=self.f_r,
            cbc_max=self.cbc_max,
            f_q=self.f_q,
            cyclen=self.cyclen,
            ao_min=self.ao_min,
            ao_max=self.ao_max,
            max_burnup=self.max_assembly_burnup,
            max_pin_burnup=self.max_pin_burnup,
            max_burnup_assembly=self.max_burnup_assembly,
            max_burnup_pin=self.max_burnup_pin,
            converged=True,
        )


@dataclass(frozen=True, slots=True)
class MasterRunResult:
    """Result metadata returned by :meth:`MasterRunner.run`."""

    metrics: MasterMetrics
    cache_key: str
    cache_hit: bool
    work_dir: Path | None


_SHF_HEADER_RE = re.compile(
    r"^[ \t]*%LPD_SHF\b[^\r\n]*(?P<eol>\r\n|\n|\r)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_NEXT_CARD_RE = re.compile(r"^[ \t]*%[^\s\r\n]+[^\r\n]*", flags=re.MULTILINE)


def _lpd_shf_span(deck: str) -> tuple[int, int, str]:
    matches = list(_SHF_HEADER_RE.finditer(deck))
    if not matches:
        raise DeckFormatError("deck does not contain a %LPD_SHF card")
    if len(matches) != 1:
        raise DeckFormatError("deck must contain exactly one %LPD_SHF card")
    header = matches[0]
    body_start = header.end()
    next_card = _NEXT_CARD_RE.search(deck, body_start)
    body_end = next_card.start() if next_card is not None else len(deck)
    if body_start == body_end:
        raise DeckFormatError("%LPD_SHF has an empty body")
    return body_start, body_end, header.group("eol")


def extract_lpd_shf(deck: str) -> str:
    """Extract the exact ``%LPD_SHF`` body, including its final line ending.

    The header and the following MASTER card are not included.  Returning the
    body exactly makes ``replace_lpd_shf(deck, extract_lpd_shf(deck))`` a byte-
    for-character round trip for the packaged decks.
    """

    body_start, body_end, _ = _lpd_shf_span(deck)
    return deck[body_start:body_end]


def replace_lpd_shf(deck: str, quarter_shf: str) -> str:
    """Replace only the body of ``%LPD_SHF`` with a nine-line quarter pattern.

    Line endings inside the supplied body are normalized to the deck's line
    ending.  The deck prefix (including the ``%LPD_SHF`` header) and suffix are
    concatenated unchanged.  Blank lines and nested MASTER cards are rejected
    so a malformed replacement cannot consume following input cards.
    """

    lines = quarter_shf.splitlines()
    if len(lines) != 9:
        raise DeckFormatError(
            f"quarter %LPD_SHF must contain exactly 9 lines, received {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise DeckFormatError("quarter %LPD_SHF may not contain blank lines")
    if any(line.lstrip().startswith("%") for line in lines):
        raise DeckFormatError("quarter %LPD_SHF may not contain MASTER card headers")

    body_start, body_end, eol = _lpd_shf_span(deck)
    replacement = eol.join(lines)
    old_body = deck[body_start:body_end]
    if body_end < len(deck) or old_body.endswith(("\r\n", "\n", "\r")):
        replacement += eol
    return deck[:body_start] + replacement + deck[body_end:]


def _header_tokens(line: str) -> list[str]:
    return [token.upper() for token in line.split()]


def _find_header(
    lines: Sequence[str], required: tuple[str, ...], edit_name: str
) -> tuple[int, list[str]]:
    for index, line in enumerate(lines):
        tokens = _header_tokens(line)
        if len(tokens) < 3 or tokens[0] != "NO." or tokens[1:3] != ["DAY", "EFPD"]:
            continue
        if all(column in tokens for column in required):
            return index, tokens
    raise SummaryParseError(f"MAS_SUM {edit_name} header anchor was not found")


def _parse_float(token: str, *, context: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise SummaryParseError(f"invalid numeric value {token!r} in {context}") from exc
    if not math.isfinite(value):
        raise SummaryParseError(f"non-finite numeric value {token!r} in {context}")
    return value


def _parse_table(
    lines: Sequence[str],
    *,
    required_columns: tuple[str, ...],
    edit_name: str,
) -> list[dict[str, float]]:
    header_index, header = _find_header(lines, required_columns, edit_name)
    indices = {column: header.index(column) for column in required_columns}
    maximum_index = max(indices.values())
    rows: list[dict[str, float]] = []
    started = False
    saw_blank_after_data = False

    for line_number in range(header_index + 1, len(lines)):
        stripped = lines[line_number].strip()
        if not stripped:
            if started:
                saw_blank_after_data = True
            continue
        if started and saw_blank_after_data:
            break

        tokens = stripped.split()
        if not tokens or re.fullmatch(r"[+-]?\d+", tokens[0]) is None:
            if started:
                break
            continue
        if len(tokens) <= maximum_index:
            raise SummaryParseError(
                f"truncated numeric row in {edit_name} at line {line_number + 1}"
            )

        context = f"{edit_name} line {line_number + 1}"
        row = {
            column: _parse_float(tokens[column_index], context=context)
            for column, column_index in indices.items()
        }
        rows.append(row)
        started = True

    if not rows:
        raise SummaryParseError(f"MAS_SUM {edit_name} contains no numeric rows")
    return rows


def _same_efpd(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9)


def _collapse_duplicate_efpd(
    rows: list[dict[str, float]], *, edit_name: str
) -> list[dict[str, float]]:
    """Collapse consecutive duplicate EFPDs and validate the resulting grid.

    MASTER can emit repeated search rows at BOC as well as at EOC.  The first
    physical row in each group is retained, preserving the parser's existing
    treatment of the trailing keff-search duplicate.
    """

    unique: list[dict[str, float]] = []
    for row in rows:
        efpd = row["EFPD"]
        if unique and _same_efpd(efpd, unique[-1]["EFPD"]):
            continue
        if unique and efpd < unique[-1]["EFPD"]:
            raise SummaryParseError(
                f"{edit_name} EFPD must be non-decreasing; "
                "only consecutive duplicate search rows are allowed"
            )
        unique.append(row)
    return unique


def _validate_metrics(metrics: MasterMetrics) -> None:
    for name, value in metrics.as_dict().items():
        if value is None or name in {"max_burnup_assembly", "max_burnup_pin"}:
            continue
        if name == "n_steps":
            if not isinstance(value, int) or value < 1:
                raise SummaryParseError("MASTER metrics require at least one unique step")
        elif not math.isfinite(float(value)):
            raise SummaryParseError(f"MASTER metric {name} is non-finite")


def parse_mas_sum(
    text: str, *, convergence_limit: float = DEFAULT_CONVERGENCE_LIMIT
) -> MasterMetrics:
    """Parse and strictly validate MASTER SUMMARY EDIT 2 and EDIT 3.

    Parsing is driven by the documented header anchors rather than fixed
    columns, so arbitrary whitespace and ``E``/``D`` scientific notation are
    accepted.  Consecutive duplicate search rows (observed at both BOC and
    EOC) are collapsed by retaining their first physical row.  The remaining
    EFPD grids must be identical and strictly increasing.

    Args:
        text: Complete ``MAS_SUM`` text.
        convergence_limit: Maximum allowed absolute EDIT 2 ``ERRFLX``.

    Raises:
        SummaryParseError: On missing/truncated tables, non-finite values,
            misaligned EFPD grids, or failure to meet the convergence limit.
    """

    if not math.isfinite(convergence_limit) or convergence_limit < 0.0:
        raise ValueError("convergence_limit must be finite and non-negative")
    lines = text.splitlines()
    edit2 = _parse_table(
        lines,
        required_columns=("EFPD", "PPM", "ERRFLX"),
        edit_name="EDIT 2",
    )
    edit3 = _parse_table(
        lines,
        required_columns=("EFPD", "AO", "FQP", "FRP"),
        edit_name="EDIT 3",
    )
    edit2 = _collapse_duplicate_efpd(edit2, edit_name="EDIT 2")
    edit3 = _collapse_duplicate_efpd(edit3, edit_name="EDIT 3")

    if len(edit2) != len(edit3) or any(
        not _same_efpd(row2["EFPD"], row3["EFPD"])
        for row2, row3 in zip(edit2, edit3)
    ):
        raise SummaryParseError("EDIT 2 and EDIT 3 unique EFPD grids do not match")

    final_efpd = edit2[-1]["EFPD"]
    maximum_efpd = max(row["EFPD"] for row in edit2)
    if not _same_efpd(final_efpd, maximum_efpd):
        raise SummaryParseError("last EDIT 2 EFPD is not the maximum cycle EFPD")

    from .burnup import parse_summary_max_assembly_burnup

    assembly_peak = parse_summary_max_assembly_burnup(text)
    metrics = MasterMetrics(
        cyclen=maximum_efpd,
        cbc_max=max(row["PPM"] for row in edit2),
        f_q=max(row["FQP"] for row in edit3),
        f_r=max(row["FRP"] for row in edit3),
        ao_min=min(row["AO"] for row in edit3),
        ao_max=max(row["AO"] for row in edit3),
        errflx_max=max(abs(row["ERRFLX"]) for row in edit2),
        n_steps=len(edit2),
        max_assembly_burnup=(None if assembly_peak is None else assembly_peak.value),
        max_burnup_assembly=(None if assembly_peak is None else assembly_peak.location),
    )
    _validate_metrics(metrics)
    if metrics.errflx_max > convergence_limit:
        raise SummaryParseError(
            f"MASTER did not converge: max |ERRFLX|={metrics.errflx_max:.6g} "
            f"exceeds {convergence_limit:.6g}"
        )
    return metrics


def parse_mas_sum_file(
    path: PathLike, *, convergence_limit: float = DEFAULT_CONVERGENCE_LIMIT
) -> MasterMetrics:
    """Read and parse a ``MAS_SUM`` file using tolerant text decoding."""

    return parse_mas_sum(
        _read_text(Path(path)), convergence_limit=convergence_limit
    )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding is total")


def _normalise_command(command: Command) -> tuple[str, ...]:
    if isinstance(command, (str, os.PathLike)):
        raw = [os.fspath(command)]
    else:
        raw = [os.fspath(token) for token in command]
    if not raw or not raw[0]:
        raise ValueError("executable command may not be empty")

    normalised: list[str] = []
    for token in raw:
        candidate = Path(token).expanduser()
        normalised.append(str(candidate.resolve()) if candidate.exists() else token)
    return tuple(normalised)


def _hash_file(hasher: Any, label: str, path: Path) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    hasher.update(b"\0")


class MasterRunner:
    """Stage, execute, parse, and cache one packaged MASTER case.

    A case directory is expected at ``package_root/cores/<case>/<seed>`` and
    must contain exactly one ``MAS_INP*.inp`` deck.  Its parent name selects
    ``package_root/bases/<case>``, which must contain exactly one ``MAS_RST.*``
    file.  ``package_root/lib/MAS_XSL`` and ``MAS_HFF`` are staged read-only by
    convention, though ordinary copies are used for isolation.

    The executable is supplied explicitly as a path/command or a sequence such
    as ``[python, fake_master.py]``.  Existence is checked only when execution is
    attempted, permitting parser-only use on machines without MASTER.
    """

    def __init__(
        self,
        package_root: PathLike,
        executable: Command,
        *,
        work_root: PathLike | None = None,
        cache_dir: PathLike | None = None,
        timeout: float = 300.0,
        convergence_limit: float = DEFAULT_CONVERGENCE_LIMIT,
        keep_success: bool = False,
        cpu_core: int | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(convergence_limit) or convergence_limit < 0.0:
            raise ValueError("convergence_limit must be finite and non-negative")
        if cpu_core is not None and (not isinstance(cpu_core, int) or cpu_core < 0):
            raise ValueError("cpu_core must be a non-negative integer or None")
        self.package_root = Path(package_root).expanduser().resolve()
        self.command = _normalise_command(executable)
        self.work_root = (
            Path(work_root).expanduser().resolve()
            if work_root is not None
            else Path(tempfile.gettempdir()).resolve() / "master_rl_work"
        )
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else self.work_root / "cache"
        )
        self.timeout = float(timeout)
        self.convergence_limit = float(convergence_limit)
        self.keep_success = bool(keep_success)
        self.cpu_core = cpu_core
        # Counts actual subprocess invocation attempts, not cache hits or
        # candidate-level evaluations performed by higher-level runners.
        self.process_calls = 0
        # Cost accounting (F-11): how many run() requests were answered from
        # the content-addressed metric cache without a subprocess.
        self.cache_hits = 0

    @property
    def raw_process_calls(self) -> int:
        """Number of MASTER subprocess invocation attempts made by this runner."""

        return self.process_calls

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        """Stage large immutable MASTER inputs without duplicating their bytes."""

        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _apply_cpu_affinity(self, process: subprocess.Popen[bytes]) -> dict[str, Any]:
        """Pin one MASTER subprocess to the configured logical processor."""

        if self.cpu_core is None:
            return {"requested_core": None, "applied": False}
        core = self.cpu_core
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            setter = kernel32.SetProcessAffinityMask
            setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            setter.restype = ctypes.c_int
            mask = 1 << core
            handle = ctypes.c_void_p(int(process._handle))  # type: ignore[attr-defined]
            if not setter(handle, ctypes.c_size_t(mask)):
                error = ctypes.get_last_error()
                raise OSError(error, f"SetProcessAffinityMask failed for CPU {core}")
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            getter = kernel32.GetProcessAffinityMask
            getter.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            getter.restype = ctypes.c_int
            observed = mask
            if getter(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                observed = int(process_mask.value)
            return {
                "requested_core": core,
                "requested_mask": mask,
                "observed_mask": observed,
                "applied": True,
                "pid": process.pid,
            }
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(process.pid, {core})
            observed = sorted(os.sched_getaffinity(process.pid))
            return {
                "requested_core": core,
                "observed_cores": observed,
                "applied": observed == [core],
                "pid": process.pid,
            }
        raise OSError("per-process CPU affinity is unsupported on this platform")

    @staticmethod
    def _only_file(directory: Path, pattern: str, description: str) -> Path:
        candidates = sorted(path for path in directory.glob(pattern) if path.is_file())
        if len(candidates) != 1:
            raise MasterRunError(
                f"expected exactly one {description} in {directory}, found {len(candidates)}"
            )
        return candidates[0]

    def _assets(
        self,
        case_dir: Path,
        supplied_deck: str | None,
        restart_path: PathLike | None,
    ) -> tuple[str, Path, Path, Path]:
        if not case_dir.is_dir():
            raise MasterRunError(f"case directory does not exist: {case_dir}")
        if supplied_deck is None:
            deck_path = self._only_file(case_dir, "MAS_INP*.inp", "MASTER input deck")
            deck = _read_text(deck_path)
        else:
            deck = supplied_deck

        if restart_path is None:
            case_name = case_dir.parent.name
            restart = self._only_file(
                self.package_root / "bases" / case_name,
                "MAS_RST.*",
                "base restart",
            )
        else:
            restart = Path(restart_path).expanduser().resolve()
            if not restart.is_file():
                raise MasterRunError(f"restart override does not exist: {restart}")
            if not restart.name.upper().startswith("MAS_RST."):
                raise MasterRunError(
                    f"restart override must have a MAS_RST.* basename: {restart}"
                )
        xsl = self.package_root / "lib" / "MAS_XSL"
        hff = self.package_root / "lib" / "MAS_HFF"
        for asset in (xsl, hff):
            if not asset.is_file():
                raise MasterRunError(f"required MASTER library is missing: {asset}")
        restart_reference = re.compile(
            rf"^[ \t]*(?:\S*[\\/])?{re.escape(restart.name)}"
            rf"(?=[ \t]*(?:#.*)?\r?$)",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if restart_reference.search(deck) is None:
            raise MasterRunError(
                f"deck does not contain a standalone reference to restart "
                f"{restart.name}"
            )
        return deck, restart, xsl, hff

    def _content_key(
        self, deck: str, restart: Path, xsl: Path, hff: Path
    ) -> str:
        hasher = sha256()
        hasher.update(b"master_rl.MasterRunner:v1\0")
        hasher.update(deck.encode("utf-8"))
        hasher.update(b"\0")
        _hash_file(hasher, "MAS_RST", restart)
        _hash_file(hasher, "MAS_XSL", xsl)
        _hash_file(hasher, "MAS_HFF", hff)
        hasher.update(repr(self.command).encode("utf-8"))
        hasher.update(b"\0")
        for index, token in enumerate(self.command):
            command_file = Path(token)
            if command_file.is_file():
                _hash_file(hasher, f"command[{index}]", command_file)
        hasher.update(f"errflx={self.convergence_limit:.17g}".encode("ascii"))
        return hasher.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> MasterMetrics | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != _CACHE_SCHEMA or payload.get("key") != key:
                return None
            metrics = MasterMetrics.from_dict(payload["metrics"])
            if metrics.errflx_max > self.convergence_limit:
                return None
            return metrics
        except (OSError, json.JSONDecodeError, KeyError, TypeError, SummaryParseError):
            return None

    def _store_cache(self, key: str, metrics: MasterMetrics) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _CACHE_SCHEMA,
            "key": key,
            "metrics": metrics.as_dict(),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{key}.",
                suffix=".tmp",
                dir=self.cache_dir,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary_name, self._cache_path(key))
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass

    def run(
        self,
        case_dir: PathLike,
        *,
        deck_text: str | None = None,
        quarter_shf: str | None = None,
        restart_path: PathLike | None = None,
        use_cache: bool = True,
        keep_success: bool | None = None,
    ) -> MasterRunResult:
        """Evaluate one case, retaining its isolated directory on failure.

        ``deck_text`` can override the packaged deck.  ``quarter_shf`` applies
        the exact nine-line replacement before hashing and staging.
        ``restart_path`` supplies a chained restart instead of the packaged
        base; its bytes participate in the content hash.  ``keep_success`` can
        override the constructor setting for this call.  Failed directories
        are always retained and exposed through :class:`MasterRunError`.
        """

        case_path = Path(case_dir).expanduser().resolve()
        deck, restart, xsl, hff = self._assets(
            case_path, deck_text, restart_path
        )
        if quarter_shf is not None:
            deck = replace_lpd_shf(deck, quarter_shf)
        key = self._content_key(deck, restart, xsl, hff)
        if use_cache:
            cached = self._load_cache(key)
            if cached is not None:
                self.cache_hits += 1
                return MasterRunResult(cached, key, True, None)

        self.work_root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_path.name)[:40] or "case"
        work_dir = Path(
            tempfile.mkdtemp(prefix=f"{safe_name}-{key[:10]}-", dir=self.work_root)
        ).resolve()

        try:
            (work_dir / "MAS_INP").write_bytes(deck.encode("utf-8"))
            self._link_or_copy(xsl, work_dir / "MAS_XSL")
            self._link_or_copy(hff, work_dir / "MAS_HFF")
            self._link_or_copy(restart, work_dir / restart.name)

            stdout_path = work_dir / "MASTER.stdout"
            stderr_path = work_dir / "MASTER.stderr"
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    self.process_calls += 1
                    process = subprocess.Popen(
                        self.command,
                        cwd=work_dir,
                        stdout=stdout,
                        stderr=stderr,
                        # patched: CREATE_NO_WINDOW — suppress the child console
                        # window on Windows (does not affect stdout/stderr file
                        # redirection).  Guarded by os.name; self-contained so the
                        # vendored file stays standalone.  See VENDOR_MANIFEST note.
                        **(
                            {"creationflags": subprocess.CREATE_NO_WINDOW}
                            if os.name == "nt"
                            else {}
                        ),
                    )
                    try:
                        affinity = self._apply_cpu_affinity(process)
                    except Exception:
                        process.kill()
                        process.wait()
                        raise
                    (work_dir / "CPU_AFFINITY.json").write_text(
                        json.dumps(affinity, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    try:
                        returncode = process.wait(timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise
            except subprocess.TimeoutExpired as exc:
                raise MasterRunError(
                    f"MASTER timed out after {self.timeout:g} seconds",
                    work_dir=work_dir,
                ) from exc
            except OSError as exc:
                raise MasterRunError(
                    f"could not invoke MASTER command {self.command!r}: {exc}",
                    work_dir=work_dir,
                ) from exc

            if returncode != 0:
                raise MasterRunError(
                    f"MASTER exited with status {returncode}",
                    work_dir=work_dir,
                )
            summary = work_dir / "MAS_SUM"
            if not summary.is_file():
                raise MasterRunError("MASTER did not produce MAS_SUM", work_dir=work_dir)
            try:
                metrics = parse_mas_sum_file(
                    summary, convergence_limit=self.convergence_limit
                )
            except SummaryParseError as exc:
                raise MasterRunError(str(exc), work_dir=work_dir) from exc

            ppi_files = sorted(path for path in work_dir.glob("MAS_PPI.*") if path.is_file())
            if ppi_files:
                if len(ppi_files) != 1:
                    raise MasterRunError(
                        f"MASTER produced {len(ppi_files)} PPI files; expected one",
                        work_dir=work_dir,
                    )
                if metrics.max_burnup_assembly is None:
                    raise MasterRunError(
                        "PPI exists but SUMMARY EDIT 5 has no maximum assembly location",
                        work_dir=work_dir,
                    )
                from .burnup import parse_ppi_max_pin_burnup

                location = re.fullmatch(r"([A-Z]+)(\d+)", metrics.max_burnup_assembly)
                if location is None:
                    raise MasterRunError(
                        "maximum assembly burnup location is malformed",
                        work_dir=work_dir,
                    )
                try:
                    pin_peak = parse_ppi_max_pin_burnup(
                        _read_text(ppi_files[0]),
                        location.group(1),
                        int(location.group(2)),
                    )
                except SummaryParseError as exc:
                    raise MasterRunError(str(exc), work_dir=work_dir) from exc
                metrics = replace(
                    metrics,
                    max_pin_burnup=pin_peak.value,
                    max_burnup_pin=pin_peak.location,
                )

            if use_cache:
                self._store_cache(key, metrics)

            retain_success = self.keep_success if keep_success is None else bool(keep_success)
            retained: Path | None = work_dir if retain_success else None
            if not retain_success:
                try:
                    shutil.rmtree(work_dir)
                except OSError:
                    retained = work_dir
            return MasterRunResult(metrics, key, False, retained)
        except MasterRunError:
            raise
        except Exception as exc:
            raise MasterRunError(
                f"unexpected MASTER integration failure: {exc}", work_dir=work_dir
            ) from exc


__all__ = [
    "DEFAULT_CONVERGENCE_LIMIT",
    "DeckFormatError",
    "MasterError",
    "MasterMetrics",
    "MasterRunError",
    "MasterRunResult",
    "MasterRunner",
    "SummaryParseError",
    "extract_lpd_shf",
    "parse_mas_sum",
    "parse_mas_sum_file",
    "replace_lpd_shf",
]
