"""Per-cell map-head level calibration — ``map_calibration.json`` (program §2.1).

The flatness-first objective consumes ``node_peak`` / ``map_cov`` as **levels**,
not ranks: they are divided by a scale, weighted, UCB-conservatized and compared
against a physical F_r safety gate.  The champion's map head is *optimistic* at
those levels — on the honest fold-C slice the ``node_peak`` bias is −0.147 and
the ``map_cov`` bias −0.058 — and the only pessimism in the acquisition is
``risk_z x ensemble spread``, which is an epistemic *disagreement* statistic and
structurally cannot express an extrapolation bias every member shares.

Program §2.1 therefore makes this artifact a **precondition** for running the
objective, and decision D1 makes the F_r safety gate read from it::

    gate = 1.70 - fr_bias - 0.5 * fr_sigma        (correction available)
    gate = 1.70                                    (no correction -> HOLD)

This module owns the artifact's schema, its loader and its resolution rules.  The
FIT lives in :mod:`..tools.fit_map_calibration` (it needs torch and the store);
nothing here imports a model, so a campaign, a report or a test can read the
artifact for the cost of one ``json.loads``.

Sign conventions (one place, because a flipped sign here is a silent licensing
error)
------------------------------------------------------------------------------
``bias`` is always ``median(pred - actual)`` — the same convention as
:mod:`..model.cell_calibrate`, so a NEGATIVE bias means the head UNDER-predicts
(is optimistic about a quantity that is bad when high).  A prediction is
de-biased by SUBTRACTING it::

    corrected = pred - bias

The gate keys are the one derived quantity: ``fr_bias`` is the **gate shift**
``max(0, -f_r.bias)``, i.e. how far the gate must move DOWN so that a candidate
passing on the predicted F_r also passes on the expected true F_r.  It is clamped
at zero because a licensing-adjacent safety gate may be TIGHTENED by a fitted
bias but must never be LOOSENED by one — that is exactly the draft 1.75 move
program §2.1 rejected.  :func:`gate_from` re-applies the clamp on read, so a
hand-edited artifact cannot loosen the gate either.

``sigma_extra`` is the dispersion the ensemble does NOT have::

    sigma_extra = sqrt(max(0, sigma_resid^2 - mean(sigma_ens^2)))
    sigma_cal   = sqrt(sigma_ens^2 + sigma_extra^2)

so the calibrated sigma is never smaller than the raw ensemble spread, still
varies candidate-by-candidate (the ensemble term is kept), and carries the
irreducible floor the spread misses.  When the ensemble is already over-dispersed
in a cell, ``sigma_extra`` is 0 and the calibrated sigma is byte-identical to the
raw one — the correction can only ever add pessimism.
"""

from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Artifact file name, read from the STORE dir (it is a property of the label
#: corpus + one champion, not of a run).  Mirrored by
#: :data:`..data.flat_scale.GATE_CALIBRATION_NAME`, which must stay equal.
ARTIFACT_NAME = "map_calibration.json"

#: Schema tag stamped into the artifact.  Bump on a breaking format change; a
#: doc carrying THIS tag is treated as machine-produced and is held to the full
#: provenance contract (see :meth:`MapCalibration.require_model`).
ARTIFACT_SCHEMA = "map_calibration_v1"

#: The three calibrated targets.  ``node_peak`` / ``map_cov`` are the objective
#: levels; ``f_r`` is not an objective in this mode at all — it exists solely to
#: feed the D1 safety gate.
TARGETS: tuple[str, ...] = ("node_peak", "map_cov", "f_r")

#: Pessimism multiplier on the gate sigma (program §2.1 / D1).  The acquisition
#: imports this through :data:`..search.acquisition.FLATPOWER_GATE_K`, which is
#: pinned equal by a test.
GATE_K = 0.5

#: A cell needs this many honest-slice rows before its own bias/sigma is
#: trusted; below it the cell falls back to the global block.
MIN_CELL_ROWS = 12

#: 1 / Phi^-1(0.75) — MAD -> SD for a Gaussian.
_MAD_TO_SD = 1.4826


class MapCalibrationError(ValueError):
    """Base class for the artifact's refusals."""


class ModelMismatchError(MapCalibrationError):
    """The artifact was fitted on a DIFFERENT champion than the one serving.

    Loud on purpose (program §2.1): a level calibration is a property of one
    checkpoint's head.  Applying model A's bias/sigma to model B's predictions
    silently mis-states the safety gate and the UCB pessimism in the exact
    direction nobody would notice — the numbers stay plausible.
    """


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- #
# model identity
# --------------------------------------------------------------------------- #
def model_id(model_dir: str | Path | None) -> str:
    """The champion's identity token — its directory NAME (e.g. ``20260725_063351``).

    The name is used rather than the absolute path so that copying / mounting the
    model tree elsewhere is not mistaken for a different champion; the content
    check is :func:`model_fingerprint`.
    """
    if model_dir is None:
        return ""
    return Path(str(model_dir)).name


#: Fingerprint schema tag, mixed into the digest.  ``v1`` hashed member
#: ``meta.json`` bytes ONLY, so two checkpoints with identical metadata and
#: different WEIGHTS fingerprinted the same and :meth:`MapCalibration.require_model`
#: positively certified one as the other — the certification did not mean what it
#: said.  ``v2`` covers ``model.pt`` as well; the tag is in the hash so a v1 token
#: can never accidentally equal a v2 one.
FINGERPRINT_SCHEMA = "v2"

#: ``(abs path, size, mtime_ns) -> sha1`` of one member's ``model.pt``.
#: The champion is 10.35M params x 5 members (~200 MB), so a full re-hash on every
#: ``require_model`` call would be paid at construction, at every resume and at
#: every per-wave champion swap.  The identity triple invalidates itself whenever
#: the file is rewritten, so the cache can never certify stale bytes.
_WEIGHTS_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}

#: Read size for the weights hash (1 MiB) — bounded memory over a 40 MB member.
_HASH_CHUNK = 1 << 20


def _weights_digest(path: Path) -> str | None:
    """SHA1 of ``model.pt``'s bytes, memoized on ``(path, size, mtime_ns)``.

    ``None`` when the file is absent or unreadable — the caller decides what an
    absent weights file means (a meta-only member fingerprints on its meta, an
    unreadable one voids the whole fingerprint).
    """
    try:
        st = path.stat()
    except OSError:
        return None
    try:
        key = (os.path.normcase(str(path.resolve())), int(st.st_size),
               int(st.st_mtime_ns))
    except (OSError, ValueError):            # pragma: no cover - exotic path
        return None
    hit = _WEIGHTS_DIGEST_CACHE.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha1()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                h.update(chunk)
    except OSError:                          # pragma: no cover - unreadable member
        return None
    digest = h.hexdigest()
    _WEIGHTS_DIGEST_CACHE[key] = digest
    return digest


def model_fingerprint(model_dir: str | Path | None) -> str:
    """Content fingerprint of an ensemble dir (``""`` when it cannot be read).

    SHA1 over each member's ``meta.json`` bytes **and its ``model.pt`` weights**,
    in sorted member order.  A retrain that reuses the SAME directory name changes
    every member's meta (``seed`` / ``best_epoch`` / ``target_zscore`` /
    ``map_zscore``), so the fingerprint catches the one case :func:`model_id`
    cannot — but metadata alone does not identify a CHECKPOINT: a fine-tune, a
    re-export or a hand-swapped ``model.pt`` leaves every meta byte-identical
    while the head that produced the fitted bias/sigma is gone.  Hashing meta only
    therefore CERTIFIED such a swap as the same model, which is precisely the
    silent misapplication :class:`ModelMismatchError` exists to prevent.

    The weights hash is memoized (:func:`_weights_digest`), so the ~200 MB
    champion is read once per (file, size, mtime) and every later call — resume,
    per-wave champion swap — is a dict lookup.
    """
    if model_dir is None:
        return ""
    root = Path(str(model_dir))
    metas = sorted(root.glob("member_*/meta.json"))
    if not metas:
        return ""
    h = hashlib.sha1()
    h.update(FINGERPRINT_SCHEMA.encode("ascii"))
    for p in metas:
        try:
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
        except OSError:                      # pragma: no cover - unreadable member
            return ""
        weights = p.with_name("model.pt")
        digest = _weights_digest(weights)
        if digest is None and weights.exists():
            # present but unreadable: refuse to fingerprint rather than fall back
            # to the meta-only digest that would certify unknown weights.
            return ""                        # pragma: no cover - unreadable member
        h.update(b"|weights:")
        h.update((digest or "absent").encode("ascii"))
    return h.hexdigest()


def _same_path(a: str, b: str) -> bool:
    """Case/separator-insensitive path equality (Windows store paths)."""
    if not a or not b:
        return False
    try:
        pa, pb = Path(a).expanduser(), Path(b).expanduser()
        return os.path.normcase(str(pa)) == os.path.normcase(str(pb))
    except (OSError, ValueError):            # pragma: no cover - exotic path
        return False


# --------------------------------------------------------------------------- #
# one fitted (cell, target) entry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetCalibration:
    """One target's measured level bias + dispersion on the honest slice."""

    bias: float = 0.0
    #: robust SD of the residual ``pred - actual`` (MAD-scaled).
    sigma: float = 0.0
    #: RMS of the per-row ENSEMBLE spread the model itself reported.
    sigma_ens: float = 0.0
    #: dispersion the ensemble does not have (``>= 0``).
    sigma_extra: float = 0.0
    n: int = 0

    @classmethod
    def from_entry(cls, entry: Any) -> "TargetCalibration | None":
        if not isinstance(entry, _MappingABC):
            return None
        bias = _finite(entry.get("bias"))
        if bias is None:
            return None
        return cls(
            bias=bias,
            sigma=max(0.0, _finite(entry.get("sigma")) or 0.0),
            sigma_ens=max(0.0, _finite(entry.get("sigma_ens")) or 0.0),
            sigma_extra=max(0.0, _finite(entry.get("sigma_extra")) or 0.0),
            n=int(entry.get("n") or 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"bias": round(float(self.bias), 6),
                "sigma": round(float(self.sigma), 6),
                "sigma_ens": round(float(self.sigma_ens), 6),
                "sigma_extra": round(float(self.sigma_extra), 6),
                "n": int(self.n)}


#: What a resolution came from — used in logs so a global fallback is visible.
SOURCE_CELL = "cell"
SOURCE_GLOBAL = "global"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class Resolved:
    """A resolved calibration plus WHERE it came from."""

    calibration: TargetCalibration | None
    source: str = SOURCE_NONE

    @property
    def available(self) -> bool:
        return self.calibration is not None

    @property
    def bias(self) -> float:
        return float(self.calibration.bias) if self.calibration else 0.0

    @property
    def sigma_extra(self) -> float:
        return float(self.calibration.sigma_extra) if self.calibration else 0.0


def gate_shift(bias: float | None) -> float:
    """``fr_bias`` (the gate shift) from an ``f_r`` level bias.

    ``max(0, -bias)``: a head that UNDER-predicts F_r (negative bias) moves the
    gate DOWN by exactly the shortfall; a head that OVER-predicts leaves it where
    it is.  The clamp is the licensing rule — see the module header.
    """
    b = _finite(bias)
    if b is None:
        return 0.0
    return max(0.0, -b)


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapCalibration:
    """A parsed ``map_calibration.json`` (or the inert empty one).

    ``cells`` maps a :func:`..model.cell_calibrate.cyclen_cell_key` cell to its
    per-target calibrations; ``globals_`` is the pooled fallback for a cell the
    fit never reached.  Everything degrades to "no correction": an absent file, a
    malformed file, an unfitted cell with no global block — all yield
    :data:`SOURCE_NONE`, which every consumer must read as "hold the uncorrected
    behaviour", never as "correction of zero is proven".
    """

    cells: Mapping[str, Mapping[str, TargetCalibration]] = field(default_factory=dict)
    globals_: Mapping[str, TargetCalibration] = field(default_factory=dict)
    fit: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ""
    source: str = "absent"
    #: Raw parsed doc — only for consumers that need a key this class does not model.
    doc: Mapping[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def empty(cls, source: str = "absent") -> "MapCalibration":
        return cls(source=source)

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any] | None, *,
                 source: str = "doc") -> "MapCalibration":
        if not isinstance(doc, _MappingABC) or not doc:
            return cls.empty(source=source)
        cells: dict[str, dict[str, TargetCalibration]] = {}
        for key, entry in (doc.get("cells") or {}).items():
            if not isinstance(entry, _MappingABC):
                continue
            per_target: dict[str, TargetCalibration] = {}
            for target in TARGETS:
                tc = TargetCalibration.from_entry(entry.get(target))
                if tc is not None:
                    per_target[target] = tc
            # A cell that carries only the derived gate keys (a hand-written
            # stub) still resolves the gate — see ``gate_for``.
            cells[str(key)] = per_target
        globals_: dict[str, TargetCalibration] = {}
        for target in TARGETS:
            tc = TargetCalibration.from_entry((doc.get("global") or {}).get(target))
            if tc is not None:
                globals_[target] = tc
        return cls(cells=cells, globals_=globals_,
                   fit=dict(doc.get("fit") or {}),
                   schema=str(doc.get("schema") or ""),
                   source=source, doc=dict(doc))

    @classmethod
    def from_store(cls, store_dir: str | Path | None) -> "MapCalibration":
        """Load ``<store_dir>/map_calibration.json``; inert when absent/unreadable.

        Absence is NOT an error here — it is the "no correction available" state
        the D1 gate is defined for.  What IS an error is a present artifact fitted
        on another champion, and that is :meth:`require_model`'s job.
        """
        if store_dir is None:
            return cls.empty()
        path = Path(store_dir) / ARTIFACT_NAME
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return cls.empty(source=f"absent ({path})")
        return cls.from_doc(doc, source=str(path))

    # -- provenance --------------------------------------------------------- #
    @property
    def present(self) -> bool:
        return bool(self.cells or self.globals_ or self.doc)

    @property
    def fit_model_id(self) -> str:
        mid = str(self.fit.get("model_id") or "")
        if mid:
            return mid
        return model_id(self.fit.get("model_dir")) if self.fit.get("model_dir") else ""

    @property
    def fit_model_fingerprint(self) -> str:
        return str(self.fit.get("model_fingerprint") or "")

    @property
    def verifiable(self) -> bool:
        """True when the artifact says which champion it was fitted on."""
        return bool(self.fit_model_id or self.fit.get("model_dir"))

    def require_model(self, model_dir: str | Path | None, *,
                      log: Any = None) -> bool:
        """Refuse a calibration fitted on a different champion (task item 4).

        Returns ``True`` when the artifact is proven to belong to ``model_dir``,
        ``False`` when there is nothing to check (no artifact, or an unverifiable
        legacy/hand-written one — in which case a warning goes to ``log``, so the
        situation is reported rather than assumed).  Raises
        :class:`ModelMismatchError` when the artifact NAMES a champion and it is
        not the serving one.

        Identity is decided by CONTENT, with the name as a hint:

        * **fingerprint** — SHA1 over the members' ``meta.json`` AND their
          ``model.pt`` weights (:func:`model_fingerprint`).  When both sides carry
          one it is the whole
          verdict: equal fingerprints are the same checkpoint however the
          directory is named (a renamed / relocated / re-saved champion passes,
          with a NOTE when the names disagree), and different fingerprints are a
          refusal however well the names match (the retrain-in-place case).
        * **name** — the model dir NAME (``20260725_063351``), or an explicitly
          recorded absolute path.  A HINT, used only when a fingerprint is
          missing on either side and there is nothing else to decide on.

        Checking the name FIRST inverted this: a byte-identical champion under a
        different path was refused before the fingerprint was ever computed, and
        — the dangerous half — the name was silently doing the job the
        fingerprint exists to do.
        """
        if not self.present:
            return False
        if not self.verifiable:
            if log is not None:
                log(f"[map_calibration] WARNING: {self.source} declares no fit "
                    "model — cannot verify it was fitted on the serving champion "
                    "(refit with lpopt.tools.fit_map_calibration)")
            return False
        if model_dir is None:
            raise ModelMismatchError(
                f"{ARTIFACT_NAME} was fitted on model {self.fit_model_id!r} but "
                "the serving champion is unknown; refusing to apply a level "
                "calibration that cannot be attributed to the serving model")

        want_id = model_id(model_dir)
        got_id = self.fit_model_id
        same_name = (not got_id) or got_id == want_id or _same_path(
            str(self.fit.get("model_dir") or ""), str(model_dir))
        want_fp = model_fingerprint(model_dir)
        got_fp = self.fit_model_fingerprint
        refit = ("A map-head level calibration is a property of ONE checkpoint: "
                 "applying it across models mis-states the F_r safety gate and "
                 "the UCB pessimism silently. Refit with `python -m "
                 "lpopt.tools.fit_map_calibration` against the serving champion, "
                 "or remove the artifact to hold the uncorrected gate.")

        if want_fp and got_fp:
            # CONTENT decides.  The name only colours the message.
            if want_fp == got_fp:
                if not same_name and log is not None:
                    log(f"[map_calibration] NOTE: {self.source} names champion "
                        f"{got_id!r} while the serving champion dir is "
                        f"{want_id!r}; both fingerprint as {want_fp[:12]}..., so "
                        "this is the SAME checkpoint under another path — "
                        "accepted on content, not on its name.")
                return True
            raise ModelMismatchError(
                f"{ARTIFACT_NAME} ({self.source}) was fitted on champion "
                f"{got_id or '<unnamed>'!r} fingerprinting {got_fp[:12]}... but "
                f"the serving champion {want_id!r} fingerprints as "
                f"{want_fp[:12]}.... "
                + ("The directory name matches but its members do not — the "
                   "champion was retrained in place. " if same_name else "")
                + refit)

        # No fingerprint on one side: the NAME is all that is left, so it decides
        # — and says so, because a name is provenance, not proof.
        if not same_name:
            raise ModelMismatchError(
                f"{ARTIFACT_NAME} ({self.source}) was fitted on champion "
                f"{got_id!r} but the serving champion is {want_id!r}, and no "
                "fingerprint is recorded on both sides to settle it by content. "
                + refit)
        return True

    # -- resolution --------------------------------------------------------- #
    def has_cell(self, cell_key: str | None) -> bool:
        return bool(cell_key and str(cell_key) in self.cells)

    def resolve(self, target: str, cell_key: str | None) -> Resolved:
        """Per-cell calibration for ``target``, else the global fallback, else none."""
        if cell_key:
            entry = self.cells.get(str(cell_key))
            if entry:
                tc = entry.get(str(target))
                if tc is not None:
                    return Resolved(tc, SOURCE_CELL)
        tc = self.globals_.get(str(target))
        if tc is not None:
            return Resolved(tc, SOURCE_GLOBAL)
        return Resolved(None, SOURCE_NONE)

    def bias(self, target: str, cell_key: str | None) -> float | None:
        """``median(pred - actual)`` for ``target`` (``None`` when unavailable)."""
        r = self.resolve(target, cell_key)
        return r.bias if r.available else None

    def sigma_extra(self, target: str, cell_key: str | None) -> float | None:
        """Ensemble-missing dispersion for ``target`` (``None`` when unavailable)."""
        r = self.resolve(target, cell_key)
        return r.sigma_extra if r.available else None

    def calibrated_sigma(self, target: str, cell_key: str | None,
                         sigma_ens: Any) -> np.ndarray:
        """``sqrt(sigma_ens^2 + sigma_extra^2)`` — never below the raw spread.

        Vectorized and NaN-preserving.  With no calibration available this is the
        identity on ``sigma_ens``, so an absent artifact keeps the raw ensemble
        behaviour exactly.
        """
        s = np.asarray(sigma_ens, dtype=float)
        extra = self.sigma_extra(target, cell_key)
        if not extra:
            return s
        return np.sqrt(s * s + float(extra) ** 2)

    # -- the D1 safety gate ------------------------------------------------- #
    def gate_for(self, cell_key: str | None) -> tuple[float | None, float | None]:
        """``(fr_bias, fr_sigma)`` for the D1 gate, or ``(None, None)``.

        Resolution order: the cell's own explicit ``fr_bias`` / ``fr_sigma`` keys,
        then the cell's fitted ``f_r`` block (``fr_bias = max(0, -bias)``,
        ``fr_sigma = sigma_extra``), then the same two at the global level.  The
        first level that yields a bias wins; anything else returns ``(None, None)``
        and the caller HOLDS the uncorrected gate.
        """
        for scope in ("cell", "global"):
            if scope == "cell":
                if not cell_key:
                    continue
                raw = ((self.doc.get("cells") or {}).get(str(cell_key)) or {})
                fitted = self.cells.get(str(cell_key), {}).get("f_r")
            else:
                raw = (self.doc.get("global") or {})
                fitted = self.globals_.get("f_r")
            if not isinstance(raw, _MappingABC):
                raw = {}
            bias = _finite(raw.get("fr_bias"))
            if bias is not None:
                return (max(0.0, bias), _finite(raw.get("fr_sigma")))
            if fitted is not None:
                return (gate_shift(fitted.bias), float(fitted.sigma_extra))
        return (None, None)

    # -- reporting ---------------------------------------------------------- #
    def describe(self, cell_key: str | None) -> dict[str, Any]:
        """What a campaign in ``cell_key`` will actually apply (for the run log)."""
        out: dict[str, Any] = {
            "artifact": self.source,
            "present": self.present,
            "cell": cell_key,
            "fitted_cell": self.has_cell(cell_key),
            "n_cells": len(self.cells),
            "fit_model_id": self.fit_model_id,
            "verifiable": self.verifiable,
        }
        for target in TARGETS:
            r = self.resolve(target, cell_key)
            out[target] = {
                "source": r.source,
                "bias": round(r.bias, 6) if r.available else None,
                "sigma_extra": round(r.sigma_extra, 6) if r.available else None,
            }
        fr_bias, fr_sigma = self.gate_for(cell_key)
        out["gate"] = {"fr_bias": fr_bias, "fr_sigma": fr_sigma,
                       "k": GATE_K, "applied": fr_bias is not None}
        return out


# --------------------------------------------------------------------------- #
# module-level convenience (the flat_scale gate helper delegates here)
# --------------------------------------------------------------------------- #
def load_gate_correction(store_dir: str | Path | None, cell_key: str | None
                         ) -> tuple[float | None, float | None]:
    """``(fr_bias, fr_sigma)`` from ``<store_dir>/map_calibration.json``.

    The D1 "is a correction available" test.  Deliberately total: absent,
    unparseable, unfitted-cell-with-no-global all return ``(None, None)`` so the
    gate HOLDS at its uncorrected value.  Note this reader performs NO model
    check — the campaign calls :meth:`MapCalibration.require_model` separately,
    because a reader that raises would turn a report into a crash.
    """
    if store_dir is None or not cell_key:
        return (None, None)
    return MapCalibration.from_store(store_dir).gate_for(cell_key)


__all__ = [
    "ARTIFACT_NAME",
    "ARTIFACT_SCHEMA",
    "FINGERPRINT_SCHEMA",
    "GATE_K",
    "MIN_CELL_ROWS",
    "SOURCE_CELL",
    "SOURCE_GLOBAL",
    "SOURCE_NONE",
    "TARGETS",
    "MapCalibration",
    "MapCalibrationError",
    "ModelMismatchError",
    "Resolved",
    "TargetCalibration",
    "gate_shift",
    "load_gate_correction",
    "model_fingerprint",
    "model_id",
]
