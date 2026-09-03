"""Serve a move policy at PROPOSAL time (results report section 7).

:class:`MoveScorer` is v1; :class:`MoveScorerV2` is v2 and differs from it in
exactly two places — the feature builder it binds and the schema stamp it
refuses to load without.  Everything else below is shared, deliberately, so the
two cannot drift apart in the serving path.

One object, one method::

    scorer.score(parent, children, ctx) -> np.ndarray[n, 2]   # P(fr), P(flat)

``parent`` is a ``(genome, pattern)`` pair and ``children`` a sequence of them —
exactly what :func:`lpopt.search.construct.build_pool` already holds at its
``mutate`` call, so nothing is decoded twice.

Two disciplines govern this module and neither is negotiable:

**The features must be the TRAINING features, not a second implementation.**
Every descriptor is produced by the code that built the corpus:
:func:`lpopt.policy.data.scalar_features` renders the 36-column move vector from
a frame whose columns are filled by ``mine_policy_corpus``'s own
:func:`classify_move` / :func:`board_physics` / :func:`_direction`, and the board
tensor comes from the same :class:`~lpopt.model.featurize.FeatureEncoder` under
the same ``v6b`` schema and power-prior constants.  The checkpoint's
``scalar_names`` are asserted against the rendered names at load, so a drift is a
load-time error rather than a silently reordered feature vector.

**The policy is a RANKER, never a probability** (report section 5: on an unseen
era its ECE is 0.111/0.200 while its parent-blocked ranking still carries
signal).  Callers may order candidates by these numbers and must not threshold
them.  It is also validated only WITHIN a parent (parent-blocked AUC 0.771) — the
pooled number is confounded — so :meth:`score` takes ONE parent and refuses to be
handed a mixed batch.

Cost: one encoder call per distinct board.  The parent is encoded once and cached,
so an ``n``-candidate expansion is ``n`` encodes plus one batched forward over the
5-member CNN ensemble.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import (
    COND_SCHEMA, HEADS, POWER_PRIOR, _grid_scatter, corpus_provenance,
    scalar_features,
)
from .v2 import CURRENT_ERA_LIBRARIES, POLICY_SCHEMA_V2, scalar_features_v2
from .v3 import (
    HEADS_V3, POLICY_SCHEMA_V3, provenance_v3, scalar_features_v3,
)

#: Where ``train_policy_v1.py`` pulled the checkpoints to.
DEFAULT_MODEL_DIR = "data/models/policy_v1"
#: Where ``train_policy_v2.py`` pulled the Run B (shipped) checkpoints to.
DEFAULT_MODEL_DIR_V2 = "data/models/policy_v2"
#: Where ``train_policy_v3.py`` pulls the v3 checkpoints to.
DEFAULT_MODEL_DIR_V3 = "data/models/policy_v3"
#: The CNN arm is the one that passed the gate; the ``mlp`` control failed it
#: outright (report section 3) and is never served.
MEMBER_PATTERN = "cnn_seed*"
FUEL_TYPES_PARQUET = "data/store/fuel_types.parquet"

#: Head index in the network's 2-logit output, from :data:`lpopt.policy.data.HEADS`.
HEAD_INDEX: dict[str, int] = {name: i for i, name in enumerate(HEADS)}
#: Head index in v3's 3-logit output.  ``fr`` / ``flat`` keep index 0 / 1, so a
#: caller that asks for ``fr`` gets the same head from either ensemble and only
#: ``fxy`` is new.
HEAD_INDEX_V3: dict[str, int] = {name: i for i, name in enumerate(HEADS_V3)}


# --------------------------------------------------------------------------- #
# the corpus miner's descriptor functions, imported BY PATH
# --------------------------------------------------------------------------- #
def _corpus() -> Any:
    """``mine_policy_corpus`` as a module, importable from any working directory.

    It is a repo-ROOT script rather than a package module, so a plain
    ``import mine_policy_corpus`` only works when the process happens to have
    been started from the repo root.  Re-deriving its ~150 lines of descriptor
    arithmetic here instead would be a second implementation of the training
    features, and the first divergence would be silent — a mis-scored move, not
    a crash.  So it is loaded by absolute path.
    """
    mod = sys.modules.get("mine_policy_corpus")
    if mod is not None:
        return mod
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "mine_policy_corpus", root / "mine_policy_corpus.py")
    if spec is None or spec.loader is None:      # pragma: no cover - defensive
        raise ImportError(f"cannot load mine_policy_corpus from {root}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mine_policy_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #
class MoveScorer:
    """A loaded 5-member policy ensemble that scores edits from ONE parent."""

    #: Serving identity.  ``get_scorer`` selects the class, the campaign readout
    #: records this string, so a wave artifact always names the model that ranked
    #: it — the fail-open failure P1-05 describes is a readout that cannot.
    version = "v1"
    #: Default checkpoint directory and member glob for THIS version.
    DEFAULT_DIR = DEFAULT_MODEL_DIR
    MEMBERS = MEMBER_PATTERN
    #: The training feature builder.  Never re-implemented here: v1 serves v1's
    #: 36 scalars, v2 serves ``scalar_features_v2``'s 39, v3 serves
    #: ``scalar_features_v3``'s 51, and the subclass swaps this one binding
    #: rather than the scoring path.
    _scalar_features = staticmethod(scalar_features)
    #: The output heads, in logit order.  v1/v2 have two; v3 has three.
    HEADS: tuple[str, ...] = HEADS
    #: ``(dataset, sym_class)`` conditioning, the SAME map the checkpoint's own
    #: corpus was featurized with.  v1/v2 use ``data.corpus_provenance`` (their
    #: corpus took ``sym_class`` from ``library_provenance``); v3 re-mined with
    #: the store truth and therefore uses ``featurize.serve_provenance``.
    _provenance = staticmethod(corpus_provenance)

    def __init__(self, members: list[Any], encoder: Any, fuel: Any,
                 enrichment: dict[str, dict[str, float]],
                 delta_channels: list[int], scalar_names: list[str],
                 *, device: str = "cpu", cache_size: int = 512,
                 fuel_types: str | Path = FUEL_TYPES_PARQUET) -> None:
        self.members = members
        self.encoder = encoder
        self.fuel = fuel
        self.enrichment = enrichment
        self.delta = np.asarray(delta_channels, np.int32)
        self.scalar_names = list(scalar_names)
        self.fuel_types = Path(fuel_types)
        self.device = device
        self._cache_size = int(cache_size)
        self._slots: dict[str, np.ndarray] = {}
        self._globals: dict[str, np.ndarray] = {}
        rows, cols, slot = _grid_scatter()
        self._rows, self._cols, self._slot = rows, cols, slot
        self._grid_shape = (encoder.n_channels + len(self.delta) + 1, 19, 19)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def _check_meta(cls, meta: dict[str, Any], member_dir: Path) -> None:
        """Refuse a checkpoint whose stamped schema is not the one served here.

        A HARD error, never a warning: the whole point of the stamp is that a
        checkpoint from another feature contract must not reach ``score()``,
        where the layout mismatch would surface as an exception that
        ``_policy_pick`` swallows into a uniform draw.
        """
        if meta.get("cond_schema", COND_SCHEMA) != COND_SCHEMA:
            raise ValueError(
                f"{member_dir.name} was trained on cond_schema "
                f"{meta.get('cond_schema')!r}, not {COND_SCHEMA!r}")

    @classmethod
    def load(cls, model_dir: str | Path | None = None, *,
             fuel_types: str | Path = FUEL_TYPES_PARQUET,
             device: str = "cpu", n_threads: int = 0) -> "MoveScorer":
        """Load every ``cnn_seed*`` member under ``model_dir`` onto ``device``.

        ``n_threads > 0`` pins torch's intra-op thread count: this runs inside a
        campaign process that is already sharing a workstation with a MASTER
        queue, so it must not quietly grab every core.

        The provenance advisory fires HERE, not only in :func:`get_scorer`,
        because ``get_scorer`` is not the only serving construction site:
        ``data/reports/.../ablation_wave.py`` builds the v1 scorer with a bare
        ``MoveScorer.load`` and scores with it, and that file is sha-pinned — a
        campaign log that cannot be edited is exactly the one that must be told.
        :func:`_warn_if_inverted` is idempotent per version and cannot raise, so
        adding it to the load path costs at most one line per process.
        """
        _warn_if_inverted(cls)
        import torch                       # deferred: construct.py stays torch-free

        from ..data.fuel_types import FuelLibrary
        from ..model.featurize import FeatureEncoder
        from ..model.power_prior import PowerPrior
        from .net import PolicyNet, PolicyNetConfig

        if n_threads > 0:
            torch.set_num_threads(int(n_threads))

        root = Path(cls.DEFAULT_DIR if model_dir is None else model_dir)
        dirs = sorted(d for d in root.glob(cls.MEMBERS)
                      if (d / "model.pt").is_file())
        if not dirs:
            raise FileNotFoundError(
                f"no {cls.MEMBERS}/model.pt checkpoints under {root}")

        encoder = FeatureEncoder(cond_schema=COND_SCHEMA,
                                 power_prior=PowerPrior(**POWER_PRIOR))
        fuel = FuelLibrary.from_parquet(fuel_types)

        members: list[Any] = []
        meta0: dict[str, Any] | None = None
        for member_dir in dirs:
            meta = json.loads((member_dir / "meta.json").read_text())
            if meta0 is None:
                meta0 = meta
            elif (meta["delta_channels"] != meta0["delta_channels"]
                  or meta["scalar_names"] != meta0["scalar_names"]
                  or meta["net_config"] != meta0["net_config"]):
                raise ValueError(
                    f"{member_dir.name} has a different feature layout than "
                    f"{dirs[0].name}; the ensemble members must share one encoding")
            cls._check_meta(meta, member_dir)
            net = PolicyNet(PolicyNetConfig(**meta["net_config"]))
            net.load_state_dict(
                torch.load(member_dir / "model.pt", map_location="cpu",
                           weights_only=True))
            net.eval().to(device)
            members.append(net)

        assert meta0 is not None
        index = {name: i for i, name in enumerate(encoder.channels)}
        missing = [c for c in meta0["delta_channels"] if c not in index]
        if missing:
            raise ValueError(f"checkpoint delta channels absent from the "
                             f"{COND_SCHEMA} encoder: {missing}")
        delta = [index[c] for c in meta0["delta_channels"]]

        scorer = cls(members, encoder, fuel,
                     _corpus().load_enrichment(Path(fuel_types)),
                     delta, meta0["scalar_names"], device=device,
                     fuel_types=fuel_types)
        cfg = meta0["net_config"]
        if scorer._grid_shape[0] != int(cfg["in_channels"]):
            raise ValueError(
                f"board tensor has {scorer._grid_shape[0]} channels but the "
                f"checkpoint expects {cfg['in_channels']}")
        return scorer

    # -- featurization ------------------------------------------------------ #
    def _board(self, pattern: Any, ctx: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(slots[C, 69], globals[G])`` for one board, memoized by pattern.

        The provenance conditioning is :func:`~.data.corpus_provenance` — the
        SAME function :func:`~.data.build_pattern_cache` featurized the training
        corpus with, which is the whole point (2026-08-29 train/serve forensic).

        It used to be :func:`~..model.featurize.library_provenance`, the
        historical-extractor map, which predates ``dataset="P"`` and answers
        paramA -> ``("A", "rot61")``.  But ``mine_policy_corpus`` writes each
        step row's REAL store ``dataset``, and EVERY paramA corpus row carries
        ``"P"`` -> ``g_dataset_flag`` 1.0 (2,388 rows in the v2 checkpoints' own
        ``corpus_sha256`` snapshot; the live steps parquet grows every wave, so
        the invariant is the fact and the count is only ever a snapshot's), so
        every paramA proposal was scored at 0.0 against a net trained at 1.0 —
        1 of the 13 cond globals inverted, on one of the two live libraries.
        Measured on ``gate_cur`` rows of the
        checkpoints' own corpus snapshot against ``policy_v2/probs.npz``: ga80
        rows agree to 3e-5 (float16 cache rounding) both before and after, paramA
        rows were off by up to **0.087 absolute** P(improve) and now agree to
        3e-5 as well.

        Note ``sym_class`` is NOT :func:`~..model.featurize.serve_provenance`'s
        ``"rot61"``: the corpus itself derived it from ``library_provenance`` and
        so trained ga80 at ``g_sym_class`` 0.0.  Serving must keep feeding the
        shipped ``policy_v2`` checkpoint what it was trained on; correcting that
        half requires RE-MINING the corpus (see :func:`~.data.corpus_provenance`).
        **So this method, on v1/v2, IS a live serving path that still feeds the
        inverted ga80 ``sym_class``** — deliberately, and consistently with the
        checkpoint it serves, but the 2026-08-29 provenance fix is NOT closed
        here.  It closes for v3 only (:class:`MoveScorerV3` binds
        :func:`~.v3.provenance_v3` = ``serve_provenance``);
        constructing a v1/v2 scorer -- :func:`get_scorer` or a bare
        :meth:`load` -- warns once per process that it serves this map.
        """
        from ..model.featurize import RecordInputs

        key = pattern.canonical()
        hit = self._slots.get(key)
        if hit is not None:
            return hit, self._globals[key]

        library_id = str(getattr(ctx, "library_id", "ga80"))
        dataset, sym_class = self._provenance(library_id)
        # e_core is deliberately NOT passed: ``build_pattern_cache`` did not pass
        # it either, so the training globals carry the encoder's own estimate and
        # a supplied e_core here would shift g_e_core off the trained scale.
        inp = RecordInputs(pattern=key, feed=int(getattr(ctx, "feed")),
                           case_pair=str(getattr(ctx, "pair")),
                           library_id=library_id, sym_class=sym_class,
                           dataset=dataset)
        slots = self.encoder.encode_slot_matrix(inp, self.fuel)
        # The training cache stored slots as float16 and widened them back at
        # collate time, so the network never saw full float32 precision.  Round
        # through float16 here for the same reason a test fixture is not
        # "improved": serving inputs must match training inputs.
        slots = slots.astype(np.float16).astype(np.float32)
        gvec = self.encoder._encode_globals(inp, self.fuel, slots).astype(np.float32)

        if len(self._slots) >= self._cache_size:
            self._slots.clear()
            self._globals.clear()
        self._slots[key], self._globals[key] = slots, gvec
        return slots, gvec

    def move_frame(self, parent: tuple[Any, Any],
                   children: Sequence[tuple[Any, Any]], ctx: Any) -> Any:
        """The move-descriptor frame ``scalar_features`` consumes, one row/child.

        Every column is filled by ``mine_policy_corpus``'s own functions, in the
        same order ``build_steps`` fills them, so a corpus row and a proposal-time
        row for the same (parent, child) pair are the same row.
        """
        import pandas as pd

        m = _corpus()
        p_genome, p_pattern = parent
        library_id = str(getattr(ctx, "library_id", "ga80"))
        enr = self.enrichment.get(library_id)
        p_phys = m.board_physics(p_pattern.canonical(), p_genome, enr)

        rows: list[dict[str, Any]] = []
        for c_genome, c_pattern in children:
            diff = m.classify_move(p_genome, c_genome)
            if diff.swap_units is None:
                span = radius = float("nan")
            else:
                r1 = m.ORBIT_UNITS[diff.swap_units[0]].radius
                r2 = m.ORBIT_UNITS[diff.swap_units[1]].radius
                span, radius = abs(r1 - r2), 0.5 * (r1 + r2)
            c_phys = m.board_physics(c_pattern.canonical(), c_genome, enr)
            row: dict[str, Any] = {
                "move_class": diff.move_class,
                "n_unit_edits": diff.n_unit_edits,
                "n_slots_changed": p_pattern.hamming(c_pattern),
                "swap_span": span,
                "swap_radius": radius,
                "feed": int(getattr(ctx, "feed")),
                # ``single_move`` is the EDIT-COUNT inference build_steps applies
                # to every lpopt_genome row; a proposed move has no sa_log.
                "single_move": bool(
                    diff.n_unit_edits
                    <= m.SINGLE_MOVE_MAX_EDITS.get(diff.move_class, -1)),
            }
            for name in m.PHYSICS:
                row[f"parent_{name}"] = p_phys[name]
                row[f"d_{name}"] = c_phys[name] - p_phys[name]
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame["fresh_radial_dir"] = m._direction(frame["d_fresh_enr_r_center"])
        frame["burnt_periph_dir"] = m._direction(frame["d_twice_burnt_periph_share"])
        return frame

    # -- scoring ------------------------------------------------------------ #
    def score(self, parent: tuple[Any, Any],
              children: Sequence[tuple[Any, Any]], ctx: Any) -> np.ndarray:
        """``[n, 2]`` mean ensemble P(improve) for ``fr`` and ``flat``.

        ``parent`` / each child is a ``(GeneralOrbitGenome, Pattern)`` pair.  All
        children must be edits of THIS parent — the validated readout is the
        parent-blocked one, and scores from different parents are not comparable.
        """
        import torch

        children = list(children)
        if not children:
            return np.zeros((0, len(self.HEADS)), dtype=np.float64)

        scalars, names = self._scalar_features(
            self.move_frame(parent, children, ctx))
        if names != self.scalar_names:
            symmetric = sorted(set(names) ^ set(self.scalar_names))
            raise ValueError("scalar feature layout drifted from the checkpoint: "
                             f"{symmetric or 'same names, different order'}")

        p_slots, p_globals = self._board(parent[1], ctx)
        grids = np.zeros((len(children), *self._grid_shape), np.float32)
        conds = np.zeros((len(children), len(p_globals) + scalars.shape[1]),
                         np.float32)
        for i, (_, c_pattern) in enumerate(children):
            c_slots, _ = self._board(c_pattern, ctx)
            d = c_slots[self.delta] - p_slots[self.delta]
            changed = (np.abs(c_slots - p_slots).max(axis=0) > 1e-6
                       ).astype(np.float32)[None]
            vals = np.concatenate([p_slots, d, changed], axis=0)
            grids[i][:, self._rows, self._cols] = vals[:, self._slot]
            conds[i] = np.concatenate([p_globals, scalars[i]])

        cells = torch.from_numpy(grids).to(self.device)
        cond = torch.from_numpy(conds).to(self.device)
        total = np.zeros((len(children), len(self.HEADS)), np.float64)
        with torch.no_grad():
            for net in self.members:
                total += torch.sigmoid(net(cells, cond)).cpu().numpy()
        return total / len(self.members)


# --------------------------------------------------------------------------- #
# v2
# --------------------------------------------------------------------------- #
class MoveScorerV2(MoveScorer):
    """The v2 ensemble: v1's serving plumbing, v2's OWN training features.

    Train/serve parity is by CONSTRUCTION, not by inspection: the conditioning
    vector is built by :func:`lpopt.policy.v2.scalar_features_v2` — the same
    function ``train_policy_v2.py`` calls — over the same move frame
    ``mine_policy_corpus`` fills.  v2's three additions
    (``d_fresh_enr_mass``, ``parent_fresh_enr_mass``, ``era_current``) need
    nothing new from the frame: the first two are already emitted because
    ``fresh_enr_mass`` is in ``mine_policy_corpus.PHYSICS``, and the third is a
    CELL attribute read off ``ctx.library_id``.

    Why the class exists at all rather than a branch inside :class:`MoveScorer`:
    a v2 checkpoint served through v1's ``scalar_features`` raises inside
    ``score()`` on a 36-vs-39 name mismatch, and ``construct._policy_pick``
    swallows every exception by design — so the miswiring degrades to a uniform
    random draw and an A/B whose treatment arm is its own control
    (``data/reports/policy_v2_results_20260817.md`` section 8).  The
    :meth:`_check_meta` refusal below is the other half of the same guard: a
    checkpoint that is not stamped with THIS serving contract never loads.

    Note for deck authors: v2's output is a normalized clipped expected
    improvement, not v1's probability, and its spread is ~3x narrower — see
    ``[acquisition] policy_prior_temperature_v2``.
    """

    version = "v2"
    DEFAULT_DIR = DEFAULT_MODEL_DIR_V2
    _scalar_features = staticmethod(scalar_features_v2)

    @classmethod
    def _check_meta(cls, meta: dict[str, Any], member_dir: Path) -> None:
        super()._check_meta(meta, member_dir)
        stamped = (meta.get("policy_schema"), str(meta.get("policy_version", "")),
                   tuple(meta.get("era_libraries") or ()))
        wanted = (POLICY_SCHEMA_V2, cls.version, CURRENT_ERA_LIBRARIES)
        if stamped != wanted:
            raise ValueError(
                f"{member_dir.name} is stamped (policy_schema, policy_version, "
                f"era_libraries) = {stamped!r} but this serving path is "
                f"{wanted!r}; refusing to serve a checkpoint from another "
                f"feature contract")
        if not meta.get("scalar_names"):
            raise ValueError(f"{member_dir.name}/meta.json carries no "
                             f"'scalar_names' feature list")

    def move_frame(self, parent: tuple[Any, Any],
                   children: Sequence[tuple[Any, Any]], ctx: Any) -> Any:
        frame = super().move_frame(parent, children, ctx)
        # Known before any move is proposed (the deck names the library), so this
        # is a cell attribute and not provenance — v2.scalar_features_v2's own
        # rule.  Same expression as ``load_universe_v2``.
        frame["era_current"] = (
            str(getattr(ctx, "library_id", "ga80")) in CURRENT_ERA_LIBRARIES)
        return frame


# --------------------------------------------------------------------------- #
# v3
# --------------------------------------------------------------------------- #
class MoveScorerV3(MoveScorerV2):
    """The v3 ensemble: THREE logits ``(fr, flat, fxy)`` over 51 scalars.

    It EXTENDS :class:`MoveScorerV2` rather than :class:`MoveScorer` because
    v3's feature vector strictly contains v2's: ``era_current`` is added by v2's
    ``move_frame`` and ``scalar_features_v3`` calls ``scalar_features_v2``, so
    subclassing is what keeps the containment true on the serve path as well as
    on the train path.  Only the stamp check is taken from the base.

    Three differences from :class:`MoveScorerV2`, and they are the three the
    pre-registration names (``policy_v3_prereg_20260831.md`` §8-F):

    * :func:`~.v3.scalar_features_v3` — v2's 39 scalars plus the twelve Gd /
      lattice descriptors, which is why :meth:`move_frame` has to emit the new
      columns.  A move that exchanges one Gd type for another is invisible in
      v2's frame; the intervention wave measured that invisibility as a +0.0712
      F_xy effect the model could not see.
    * :attr:`_provenance` is :func:`~.v3.provenance_v3`
      (= ``featurize.serve_provenance``).  The v3 corpus was re-mined with the
      store's own provenance, so the serve path must reconstruct THAT, not the
      inverted ``sym_class`` the shipped v2 checkpoint has to keep being fed.
    * the score is ``[n, 3]``.  ``fr`` and ``flat`` stay at logit 0 and 1, so a
      caller asking for ``fr`` gets the same head from either ensemble.

    The schema stamp is refused if it is not exactly this contract, for the
    reason the v2 class documents: ``construct._policy_pick`` swallows every
    exception by design, so a mis-fed checkpoint degrades to a uniform random
    draw and an A/B whose treatment arm is its own control.
    """

    version = "v3"
    DEFAULT_DIR = DEFAULT_MODEL_DIR_V3
    HEADS = HEADS_V3
    _scalar_features = staticmethod(scalar_features_v3)
    _provenance = staticmethod(provenance_v3)

    @classmethod
    def _check_meta(cls, meta: dict[str, Any], member_dir: Path) -> None:
        # MoveScorer's, not MoveScorerV2's: the v2 stamp is exactly what must be
        # REFUSED here.
        MoveScorer._check_meta(meta, member_dir)
        stamped = (meta.get("policy_schema"), str(meta.get("policy_version", "")),
                   tuple(meta.get("era_libraries") or ()))
        wanted = (POLICY_SCHEMA_V3, cls.version, CURRENT_ERA_LIBRARIES)
        if stamped != wanted:
            raise ValueError(
                f"{member_dir.name} is stamped (policy_schema, policy_version, "
                f"era_libraries) = {stamped!r} but this serving path is "
                f"{wanted!r}; refusing to serve a checkpoint from another "
                f"feature contract")
        n_heads = int((meta.get("net_config") or {}).get("n_heads", 0))
        if n_heads != len(cls.HEADS):
            raise ValueError(
                f"{member_dir.name} has {n_heads} logits but the v3 serving "
                f"contract is {len(cls.HEADS)} {list(cls.HEADS)}")
        if not meta.get("scalar_names"):
            raise ValueError(f"{member_dir.name}/meta.json carries no "
                             f"'scalar_names' feature list")

    @property
    def fuel_table(self) -> dict[str, dict[str, dict[str, float]]]:
        """``library_id -> {type_id: {n_gd, gd_wt, kinf0}}``, loaded once.

        The corpus miner's own loader, for the same reason the enrichment table
        is: the descriptors must be built by the code that built the training
        columns, not by a second implementation whose first divergence would be
        a silently mis-scored move.
        """
        table = getattr(self, "_fuel_table", None)
        if table is None:
            table = self._fuel_table = _corpus().load_fuel_table(self.fuel_types)
        return table

    def move_frame(self, parent: tuple[Any, Any],
                   children: Sequence[tuple[Any, Any]], ctx: Any) -> Any:
        frame = super().move_frame(parent, children, ctx)
        m = _corpus()
        types = self.fuel_table.get(str(getattr(ctx, "library_id", "ga80")))
        p_pat = parent[1].canonical()
        p_gd = m.gd_profile(p_pat, types)

        rows: list[dict[str, Any]] = []
        for _c_genome, c_pattern in children:
            c_pat = c_pattern.canonical()
            c_gd = m.gd_profile(c_pat, types)
            row: dict[str, Any] = {}
            for name in m.GD_PHYSICS:
                row[f"parent_{name}"] = p_gd[name]
                row[f"child_{name}"] = c_gd[name]
                row[f"d_{name}"] = c_gd[name] - p_gd[name]
            row.update(m.fresh_type_move(p_pat, c_pat, types))
            rows.append(row)
        for col in m.V3_SCHEMA_COLUMNS:
            frame[col] = [r[col] for r in rows]
        return frame


#: Serving class per ``version`` tag.
SCORERS: dict[str, type[MoveScorer]] = {
    "v1": MoveScorer, "v2": MoveScorerV2, "v3": MoveScorerV3}


# --------------------------------------------------------------------------- #
# load-once handle
# --------------------------------------------------------------------------- #
_CACHE: dict[tuple[str, str, str, str, int], MoveScorer | None] = {}

#: Versions this process has already warned about (see :func:`_warn_if_inverted`).
_PROVENANCE_WARNED: set[str] = set()


def _warn_if_inverted(cls: type[MoveScorer]) -> None:
    """Say it out loud, once per process, when the served map is the inverted one.

    The 2026-08-29 provenance fix is closed on the SURROGATE serve path
    (``model_api._record_inputs`` -> ``featurize.serve_provenance``) and is
    deliberately OPEN here: v1/v2 bind :func:`~.data.corpus_provenance`, whose
    ``sym_class`` half is still ``library_provenance``'s, so every ga80 board is
    conditioned on ``g_sym_class`` 0.0 against store rows that say ``rot61``.
    That is the only self-consistent thing to feed the shipped ``policy_v2``
    checkpoint — but a reader of the C.3 addendum can easily carry "fixed" over
    to this path, and a campaign that ships an A/B on it should have the residual
    named in its own log rather than only in a docstring.  Warn, do not raise:
    serving the checkpoint what it trained on is correct, not a fault.

    Called from BOTH construction sites -- :func:`get_scorer` and
    :meth:`MoveScorer.load` -- because ``ablation_wave.py``'s blind A/B builds
    its v1 scorer with a bare ``load`` and is sha-pinned, so a ``get_scorer``-only
    advisory would miss the single campaign that most needs it.  Idempotent per
    version, so being reached twice costs nothing.

    The ``warn`` itself is guarded, for the reason ``construct._policy_pick``
    guards the import: under ``-W error`` a warning IS an exception, it would be
    raised inside that function's ``try``, and an ADVISORY would silently demote
    the elite arm to unscored random mutation (or abort a strict campaign).  A
    note about the model may never decide whether the model runs.
    """
    if cls._provenance is not corpus_provenance or cls.version in _PROVENANCE_WARNED:
        return
    _PROVENANCE_WARNED.add(cls.version)
    msg = (
        f"policy scorer {cls.version} serves data.corpus_provenance: ga80 boards "
        f"are conditioned on g_sym_class=0.0 ('free69', from "
        f"featurize.library_provenance) while the store's campaign ga80 rows "
        f"carry 'rot61'.  This is DELIBERATE — the v2 pattern cache was built "
        f"that way, so the shipped checkpoint must keep being fed it — and it is "
        f"NOT covered by the 2026-08-29 serve_provenance fix, which closed the "
        f"surrogate path only.  It closes when data/policy/steps.parquet is "
        f"re-mined with the store's own sym_class and a v3 checkpoint is "
        f"promoted into data/models/policy_v3 (version='v3' serves "
        f"featurize.serve_provenance today).")
    try:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except Exception:                     # noqa: BLE001 — see docstring
        print(f"[policy] WARNING {msg}", file=sys.stderr, flush=True)


def get_scorer(model_dir: str | Path | None = None, *,
               version: str = "v1",
               fuel_types: str | Path = FUEL_TYPES_PARQUET,
               device: str = "cpu", n_threads: int = 0,
               strict: bool = False) -> MoveScorer | None:
    """The process-wide scorer for ``(version, model_dir)``, or ``None``.

    Non-strict (research) it never raises.  A missing checkpoint, a missing
    torch, a fuel table that will not read — every one of them means "run without
    the prior", exactly as ``construct._score_completions`` swallows a surrogate
    failure rather than aborting construction.  The failure is cached too, so a
    broken path costs one attempt per process instead of one per wave.

    ``strict=True`` is the PRODUCTION switch (review section 6.12): the deck asked
    for a policy, so a policy that will not load is a stop, not a silent control
    arm.  The failure is then neither swallowed nor cached — the caller sees the
    original exception, with its schema-refusal message intact.
    """
    cls = SCORERS[str(version)]
    _warn_if_inverted(cls)
    root = Path(cls.DEFAULT_DIR if model_dir is None else model_dir)
    key = (str(version), str(root), str(Path(fuel_types)), str(device),
           int(n_threads))
    if key in _CACHE and not (strict and _CACHE[key] is None):
        return _CACHE[key]
    try:
        scorer: MoveScorer | None = cls.load(
            root, fuel_types=fuel_types, device=device, n_threads=n_threads)
    except Exception:                     # noqa: BLE001 — see docstring
        if strict:
            raise
        scorer = None
    _CACHE[key] = scorer
    return scorer


__all__ = ["DEFAULT_MODEL_DIR", "DEFAULT_MODEL_DIR_V2", "DEFAULT_MODEL_DIR_V3",
           "HEAD_INDEX", "HEAD_INDEX_V3", "MoveScorer", "MoveScorerV2",
           "MoveScorerV3", "SCORERS", "get_scorer"]
