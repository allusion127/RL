"""P-core discovery, hybrid-core scheduling and parallel candidate evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
from typing import Any, Iterator, Sequence

from .dataset import CaseData
from .domain import Pattern
from .search import EvaluationResult, PatternEvaluator


def detect_performance_cores() -> list[int]:
    """Return logical CPU indices in the highest Windows efficiency class.

    Windows reports heterogeneous CPU topology through
    ``GetSystemCpuSetInformation``.  Intel P-cores use the highest efficiency
    class on the target workstation.  Homogeneous/other platforms fall back
    to all logical CPUs so the execution model remains portable.
    """

    logical_count = os.cpu_count() or 1
    if os.name != "nt":
        return list(range(logical_count))

    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.GetSystemCpuSetInformation
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
        needed = ctypes.c_ulong()
        query(None, 0, ctypes.byref(needed), None, 0)
        if needed.value == 0:
            raise OSError("GetSystemCpuSetInformation returned no data")
        buffer = ctypes.create_string_buffer(needed.value)
        if not query(buffer, needed.value, ctypes.byref(needed), None, 0):
            raise ctypes.WinError(ctypes.get_last_error())

        records: list[tuple[int, int]] = []
        offset = 0
        raw = buffer.raw
        while offset + 20 <= needed.value:
            size = int.from_bytes(raw[offset : offset + 4], "little")
            record_type = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            if size < 20 or offset + size > needed.value:
                break
            if record_type == 0:  # CpuSetInformation
                logical_index = raw[offset + 14]
                efficiency_class = raw[offset + 18]
                records.append((logical_index, efficiency_class))
            offset += size
        if not records:
            raise OSError("no CPU-set topology records found")
        highest_class = max(value for _, value in records)
        cores = sorted({index for index, value in records if value == highest_class})
        return cores or list(range(logical_count))
    except (AttributeError, OSError, ValueError):
        return list(range(logical_count))


@dataclass(frozen=True)
class CoreLayout:
    """Hybrid CPU topology: MASTER owns P-cores, the host does the rest.

    On the target Arrow Lake workstation (Core Ultra 285K: 8 P-cores + 16
    E-cores, no SMT) MASTER equilibrium chains are pinned 1:1 to performance
    cores while parsing, caching, archiving and idle PPO threads run on the
    efficiency cores.  Homogeneous machines degrade gracefully: ``efficiency``
    is empty and every scheduling hint becomes a no-op.
    """

    performance: tuple[int, ...]
    efficiency: tuple[int, ...]

    @property
    def all_cores(self) -> tuple[int, ...]:
        return tuple(sorted({*self.performance, *self.efficiency}))

    @property
    def hybrid(self) -> bool:
        return bool(self.efficiency)

    def as_dict(self) -> dict[str, Any]:
        return {
            "performance": list(self.performance),
            "efficiency": list(self.efficiency),
            "logical_total": len(self.all_cores),
            "hybrid": self.hybrid,
        }


def detect_core_layout() -> CoreLayout:
    """Split logical CPUs into performance and efficiency sets."""

    performance = detect_performance_cores()
    logical = list(range(os.cpu_count() or 1))
    efficiency = [core for core in logical if core not in set(performance)]
    return CoreLayout(tuple(performance), tuple(efficiency))


def _current_process_affinity() -> tuple[int, ...] | None:
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            getter = kernel32.GetProcessAffinityMask
            getter.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            getter.restype = ctypes.c_int
            handle = kernel32.GetCurrentProcess()
            if not getter(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                return None
            mask = int(process_mask.value)
            return tuple(index for index in range(64) if mask & (1 << index))
        except (AttributeError, OSError, ValueError):
            return None
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return None


def _set_process_affinity(cores: Sequence[int]) -> bool:
    if not cores:
        return False
    # Windows processor groups: a single affinity mask addresses at most 64
    # logical CPUs.  Beyond that the shift would silently wrap, so refuse
    # instead of pinning to the wrong cores (irrelevant on the 24-CPU target).
    if os.name == "nt" and any(int(core) >= 64 for core in cores):
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            setter = kernel32.SetProcessAffinityMask
            setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            setter.restype = ctypes.c_int
            mask = 0
            for core in cores:
                mask |= 1 << int(core)
            handle = kernel32.GetCurrentProcess()
            return bool(setter(handle, ctypes.c_size_t(mask)))
        except (AttributeError, OSError, ValueError):
            return False
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(int(core) for core in cores))
            return True
        except OSError:
            return False
    return False


@contextmanager
def host_affinity(cores: Sequence[int] | None) -> Iterator[bool]:
    """Temporarily confine the host process to ``cores``.

    Used while a MASTER wave holds the P-cores: the Python host (deck
    staging, ``MAS_SUM`` parsing, archiving, live-figure redraws) moves to the
    efficiency cores so it never steals cycles from the pinned physics
    solvers.  Children spawned during the window re-pin themselves explicitly
    (``MasterRunner._apply_cpu_affinity``), which Windows permits because a
    process affinity mask only has to be a subset of the *system* mask, not of
    its parent's.  The previous affinity is always restored, even on error.
    """

    if not cores:
        yield False
        return
    previous = _current_process_affinity()
    if previous is None:
        # If the current mask cannot be read it cannot be restored either;
        # never apply a restriction that would outlive the wave.
        yield False
        return
    applied = _set_process_affinity(cores)
    try:
        yield applied
    finally:
        if applied:
            _set_process_affinity(previous)


class ParallelPatternEvaluator:
    """Evaluate one candidate per dedicated worker in fixed-size waves.

    Each worker owns its evaluator (and therefore its MASTER work/cache roots
    and CPU affinity).  Exceptions are returned in-place so one failed deck
    does not cancel the other candidates in the same physical evaluation wave.
    """

    def __init__(
        self,
        evaluators: Sequence[PatternEvaluator],
        *,
        cpu_cores: Sequence[int] | None = None,
        host_cores: Sequence[int] | None = None,
    ) -> None:
        if not evaluators:
            raise ValueError("at least one evaluator is required")
        if cpu_cores is not None and len(cpu_cores) != len(evaluators):
            raise ValueError("cpu_cores and evaluators must have equal length")
        self.evaluators = tuple(evaluators)
        self.cpu_cores = tuple(cpu_cores) if cpu_cores is not None else None
        # E-cores the host retreats to while MASTER waves own the P-cores.
        self.host_cores = tuple(host_cores) if host_cores else None
        self._round_robin = 0
        self._lock = threading.Lock()

    @property
    def worker_count(self) -> int:
        return len(self.evaluators)

    @staticmethod
    def _evaluate_one(
        evaluator: PatternEvaluator,
        case: CaseData,
        pattern: Pattern,
    ) -> EvaluationResult | Exception:
        try:
            return evaluator.evaluate(case, pattern)
        except Exception as error:
            return error

    def evaluate_many(
        self,
        case: CaseData,
        patterns: Sequence[Pattern],
    ) -> list[EvaluationResult | Exception]:
        outcomes: list[EvaluationResult | Exception] = []
        with host_affinity(self.host_cores):
            with ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="master-worker",
            ) as pool:
                for start in range(0, len(patterns), self.worker_count):
                    wave = patterns[start : start + self.worker_count]
                    futures = [
                        pool.submit(
                            self._evaluate_one, self.evaluators[index], case, pattern
                        )
                        for index, pattern in enumerate(wave)
                    ]
                    outcomes.extend(future.result() for future in futures)
        return outcomes

    def evaluate(self, case: CaseData, pattern: Pattern) -> EvaluationResult:
        with self._lock:
            index = self._round_robin % self.worker_count
            self._round_robin += 1
        outcome = self._evaluate_one(self.evaluators[index], case, pattern)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


__all__ = [
    "CoreLayout",
    "ParallelPatternEvaluator",
    "detect_core_layout",
    "detect_performance_cores",
    "host_affinity",
]
