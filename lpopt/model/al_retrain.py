"""Champion-faithful active-learning retrain harness (forensic 20260722).

AL retrains #1/#2 failed the honest gate on the SAME cell/magnitude
(``5.5-5.75_f133`` cyclen, drop 0.172) because a plain ``lpopt.remote train
--ensemble 5 --split S1`` uses DEFAULT hyperparameters — it drops the champion's
``v5_distill_w160`` recipe (width 160 + per-cell distillation w=0.3 + cyclen
physics prior + quantile heads + auto cell-calibration + promoted asm-BU), so the
crowded mid-band ranking the distillation stabilizes collapses.

This harness reconstructs the champion's exact training recipe from its persisted
``member_*/meta.json`` + ``ensemble.json`` and auto-composes the retrain
invocation, then (a) REFRESHES the distillation teacher to the CURRENT champion
and (b) folds in the boundary-F_r improvements added in the model backlog
(``f_r_rank_loss`` default-on; optional map/F_r consistency).  ``--dry-run`` prints
the fully-composed commands without executing — the real retrain #3 launches when
the boundary round-1 data arrives.

Scope: ``lpopt/model/*`` only (the search stack + fuel_types stay frozen).
Runnable as ``python -m lpopt.model.al_retrain --champion <dir> --dry-run``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# champion recipe recovery
# --------------------------------------------------------------------------- #
def _first_member_meta(model_dir: Path) -> dict:
    """The first member's ``meta.json`` (all members share the recipe)."""
    ens = model_dir / "ensemble.json"
    members: list[str] = []
    if ens.is_file():
        members = list(json.loads(ens.read_text(encoding="utf-8")).get("members", []))
    if not members:
        members = sorted(p.name for p in model_dir.glob("member_*") if p.is_dir())
    if not members:
        raise FileNotFoundError(f"no members under {model_dir}")
    meta = model_dir / members[0] / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"missing {meta}")
    return json.loads(meta.read_text(encoding="utf-8"))


def champion_recipe(model_dir: str | Path) -> dict:
    """Normalized training recipe recovered from a champion checkpoint.

    Reads ``ensemble.json`` (member count, split, base seed) + the first member's
    ``meta.json`` (``train_config`` + ``net_config``) and returns exactly the knobs
    needed to reproduce the recipe: ensemble/split, net width/blocks/head_hidden,
    and every default-diverging training switch (physics prior, quantile heads +
    weight, distillation weight/min-match, promoted asm-BU, cyclen rank).  A field
    the checkpoint predates simply reports its schema default (so an older champion
    still yields a runnable recipe).
    """
    model_dir = Path(model_dir)
    ens = {}
    if (model_dir / "ensemble.json").is_file():
        ens = json.loads((model_dir / "ensemble.json").read_text(encoding="utf-8"))
    meta = _first_member_meta(model_dir)
    tc = dict(meta.get("train_config", {}))
    net = dict(meta.get("net_config", {}))
    return {
        "model_dir": str(model_dir),
        "n_members": int(ens.get("n_members", len(ens.get("members", [])) or 5)),
        "split": str(ens.get("split", tc.get("split", "S1"))),
        "base_seed": int(ens.get("base_seed", meta.get("seed", 0)) or 0),
        "cond_schema": str(meta.get("cond_schema", "v5")),
        "target_names": list(meta.get("target_names", [])),
        # net geometry
        "width": int(net.get("width", tc.get("width", 160))),
        "n_blocks": int(net.get("n_blocks", tc.get("n_blocks", 6))),
        "head_hidden": int(net.get("head_hidden", tc.get("head_hidden", 256))),
        # training switches (the recipe fingerprint the plain retrain dropped)
        "epochs": int(tc.get("epochs", 150)),
        "cyclen_physics_prior": bool(tc.get("cyclen_physics_prior", False)),
        "quantile_heads": bool(tc.get("quantile_heads", False)),
        "quantile_weight": float(tc.get("quantile_weight", 0.2)),
        "quantile_targets": list(tc.get("quantile_targets", ["f_r", "cyclen"])),
        "distill_weight": float(tc.get("distill_weight", 0.0)),
        "distill_min_match_frac": float(tc.get("distill_min_match_frac", 0.5)),
        "distill_targets": tc.get("distill_targets"),
        "promote_max_asm_bu": bool(tc.get("promote_max_asm_bu", False)),
        # F_xy prior-residual head.  ``target_names`` WINS (it is what the weights
        # were built with), so a champion carrying the head can never be
        # "reproduced" without it — the same silent-drop class the cond_schema /
        # head_hidden fix closed.  A stale ``promote_fxy: false`` in an older
        # train_config cannot override the head that is actually in the tensor.
        "promote_fxy": (bool(tc.get("promote_fxy", False))
                        or "f_xy" in list(meta.get("target_names", []))),
        "auto_fit_cell_calibration": bool(tc.get("auto_fit_cell_calibration", True)),
        "cyclen_rank_weight": float(tc.get("cyclen_rank_weight", 0.1)),
        "num_workers": int(tc.get("num_workers", 8)),
        # hires map-path structure (arm A6, champion since 2026-07-25).  Read from
        # net_config first because THAT is what the weights were built with;
        # train_config is the fallback for a checkpoint that predates the field.
        "map_head_mode": str(net.get("map_head_mode",
                                     tc.get("map_head_mode", "linear"))),
        "map_prior_residual": bool(tc.get("map_prior_residual",
                                          int(net.get("map_prior_channel", -1)) >= 0)),
        "map_spectral_weight": float(tc.get("map_spectral_weight", 0.0)),
        "map_peak_weight": float(tc.get("map_peak_weight", 0.0)),
        # axial profile head (decision D10).  Read from net_config first — THAT is
        # what the weights were built with — so a champion carrying the head can
        # never be "reproduced" without it (the same class of silent-drop bug the
        # cond_schema / head_hidden fix above closed).
        "axial_head": bool(tc.get("axial_head",
                                  int(net.get("n_axial_modes", 0)) > 0)),
        "axial_rank": int(net.get("n_axial_modes", tc.get("axial_rank", 6)) or 6),
        "axial_weight": float(tc.get("axial_weight", 0.2)),
    }


# --------------------------------------------------------------------------- #
# invocation composition
# --------------------------------------------------------------------------- #
#: Prereg Amendment D.4 arm 4 (2026-08-31), REGISTERED constants — the trunk LR
#: multiple explicitly "is not adjusted after the fact", and the f_xy switches
#: are arm 3's, unchanged.  Changing any of them requires a new pre-registration.
ARM4_TRUNK_LR_MULT = 0.05
ARM4_FXY_SELECT_WEIGHT = 0.5
ARM4_WARMUP_EPOCHS = 2

#: Prereg Amendment E.3 (2026-09-03) arm 5 — arm 4 VERBATIM plus a within-cell
#: pairwise margin-rank hinge on the composed f_xy row.  E.3 freezes these six
#: constants *and the selection band* before the run ("값을 바꾸는 재시도는 새
#: 사전등록을 요구한다"), so changing any of them requires a new
#: pre-registration, not an edit here.  Nothing else about arm 4 moves — that is
#: what makes the comparison paired.
ARM5_FXY_RANK_WEIGHT = 3.0
ARM5_FXY_RANK_CELL = "gate"
ARM5_FXY_RANK_MARGIN_Z = 0.1
ARM5_FXY_RANK_MIN_GAP = 0.005
ARM5_FXY_RANK_LOW_THRESH = 1.60
ARM5_FXY_RANK_LOW_WEIGHT = 3.0
ARM5_FXY_SELECT_BAND = 0.50


def recipe_to_train_args(
    recipe: dict,
    *,
    distill_cache: str | Path | None,
    fr_rank_weight: float = 0.1,
    map_fr_consistency_weight: float = 0.0,
    init_from: str | Path | None = None,
    freeze_trunk_cyclen: bool = False,
    trunk_finetune_lr_mult: float = 0.0,
    fxy_prior_on_predicted: bool = False,
    fxy_select_weight: float | None = None,
    warmup_epochs: int | None = None,
    fxy_rank: bool = False,
    fxy_select_band: float | None = None,
) -> list[str]:
    """Compose the ``lpopt.model.train`` CLI args that reproduce ``recipe``.

    Mirrors every champion switch onto its flag; ADDS the model-backlog boundary
    improvements (``--f-r-rank-weight`` default-on; ``--map-fr-consistency-weight``
    when > 0).  ``distill_cache`` (the REFRESHED teacher cache) is threaded to
    ``--distill-targets`` with the champion's ``--distill-weight`` /
    ``--distill-min-match-frac``.  ``--out-dir`` is intentionally omitted — the
    remote wrapper stamps ``runs/<ts>``.  ``--no-auto-cell-calibration`` is emitted
    only when the champion had auto-calibration OFF (default is on).
    """
    args: list[str] = [
        "--ensemble", str(int(recipe["n_members"])),
        "--split", str(recipe["split"]),
        # cond_schema and head_hidden were previously READ into the recipe but
        # never emitted, so a retrain from a v6 champion silently rebuilt at the
        # trainer's v3 default with a 256-wide head — a different model wearing
        # the champion's name.  Both are now explicit.
        "--cond-schema", str(recipe.get("cond_schema", "v5")),
        "--width", str(int(recipe["width"])),
        "--n-blocks", str(int(recipe["n_blocks"])),
        "--head-hidden", str(int(recipe.get("head_hidden", 256))),
        "--epochs", str(int(recipe["epochs"])),
        "--num-workers", str(int(recipe.get("num_workers", 8))),
        "--device", "auto",
    ]
    # hires map-path structure — MUST accompany a v6 cond_schema (see
    # config.ModelSection): v6 channels behind a linear map head is strictly worse
    # than v5, since the extra channels are paid for and cannot be read.
    if str(recipe.get("map_head_mode", "linear")) != "linear":
        args += ["--map-decoder", str(recipe["map_head_mode"])]
    if recipe.get("map_prior_residual"):
        args.append("--map-prior-residual")
    if float(recipe.get("map_spectral_weight", 0.0)) > 0.0:
        args += ["--map-spectral-weight", str(float(recipe["map_spectral_weight"]))]
    if float(recipe.get("map_peak_weight", 0.0)) > 0.0:
        args += ["--map-peak-weight", str(float(recipe["map_peak_weight"]))]
    if recipe.get("axial_head"):
        args += ["--axial-head",
                 "--axial-rank", str(int(recipe.get("axial_rank", 6))),
                 "--axial-weight", str(float(recipe.get("axial_weight", 0.2)))]
    if recipe.get("cyclen_physics_prior"):
        args.append("--cyclen-physics-prior")
    if recipe.get("quantile_heads"):
        args += ["--quantile-heads", "--quantile-weight",
                 str(float(recipe["quantile_weight"]))]
    if recipe.get("promote_max_asm_bu"):
        args.append("--promote-max-asm-bu")
    if recipe.get("promote_fxy"):
        args.append("--promote-fxy")
    # Freeze-finetune: champion weights in, trunk frozen, new head row trained.
    # ``--freeze-trunk-cyclen`` / ``--trunk-finetune-lr-mult`` REQUIRE
    # ``--init-from`` (train.py errors otherwise), so they are emitted together
    # and never independently — and they are mutually exclusive, so arm 4's
    # multiplier REPLACES the freeze flag rather than joining it.
    if init_from is not None:
        args += ["--init-from", str(init_from)]
        if float(trunk_finetune_lr_mult) > 0.0:
            args += ["--trunk-finetune-lr-mult",
                     str(float(trunk_finetune_lr_mult))]
        elif freeze_trunk_cyclen:
            args.append("--freeze-trunk-cyclen")
    if distill_cache is not None and float(recipe.get("distill_weight", 0.0)) > 0.0:
        args += ["--distill-targets", str(distill_cache),
                 "--distill-weight", str(float(recipe["distill_weight"])),
                 "--distill-min-match-frac", str(float(recipe["distill_min_match_frac"]))]
    if not recipe.get("auto_fit_cell_calibration", True):
        args.append("--no-auto-cell-calibration")
    # model-backlog boundary-F_r improvements (parity_round1c_20260722 [1]).
    args += ["--f-r-rank-weight", str(float(fr_rank_weight))]
    if float(map_fr_consistency_weight) > 0.0:
        args += ["--map-fr-consistency-weight", str(float(map_fr_consistency_weight))]
    # f_xy head switches (arm 3's, carried unchanged into arm 4).  Emitted LAST so
    # the composed line is the prereg's D.4.2 command token-for-token.
    if fxy_prior_on_predicted:
        args.append("--fxy-prior-on-predicted")
    if fxy_select_weight is not None:
        args += ["--fxy-select-weight", str(float(fxy_select_weight))]
    if warmup_epochs is not None:
        args += ["--warmup-epochs", str(int(warmup_epochs))]
    # arm 5's six rank knobs + the selection band, emitted after arm 4's block so
    # the composed line is prereg E.3.2's [3] token-for-token.  The trainer
    # refuses ``--fxy-rank-weight > 0`` without ``--promote-fxy``, and arm 5
    # implies the same ``--init-from``/``--promote-fxy`` spine as arm 4.
    if fxy_rank:
        args += ["--fxy-rank-weight", str(float(ARM5_FXY_RANK_WEIGHT)),
                 "--fxy-rank-cell", str(ARM5_FXY_RANK_CELL),
                 "--fxy-rank-margin-z", str(float(ARM5_FXY_RANK_MARGIN_Z)),
                 "--fxy-rank-min-gap", str(float(ARM5_FXY_RANK_MIN_GAP)),
                 "--fxy-rank-low-thresh", str(float(ARM5_FXY_RANK_LOW_THRESH)),
                 "--fxy-rank-low-weight", str(float(ARM5_FXY_RANK_LOW_WEIGHT))]
    if fxy_select_band is not None:
        args += ["--fxy-select-band", str(float(fxy_select_band))]
    return args


def remote_invocation(train_args: list[str], *, input_deck: str = "lpopt.inp") -> str:
    """The ``lpopt.remote train`` command that forwards ``train_args`` to the GPU.

    The remote wrapper replaces its default ``[--ensemble N --split S]`` with any
    forwarded ``train_args`` (which therefore MUST carry ensemble/split), so the
    full recipe rides through after the ``--`` stop token.

    **``--input`` MUST precede the sub-command.**  ``lpopt.remote``'s parser has
    two positionals (``cmd``, then ``train_args`` with ``nargs="*"``) and argparse
    consumes positionals greedily at the first opportunity.  With an optional
    wedged between them — ``train --input X -- ...`` — argparse matches BOTH
    positionals against the single slot ahead of ``--input``, leaves
    ``train_args`` empty, and reports every forwarded flag as an unrecognized
    argument (``error: unrecognized arguments: -- --ensemble 5 ...``).  Putting
    the wrapper's own options first parses correctly.  Verified 2026-07-25 after
    this exact ordering broke an A/B launch.
    """
    fwd = " ".join(shlex.quote(a) for a in train_args)
    return (f"python -m lpopt.remote --input {shlex.quote(input_deck)} "
            f"train -- {fwd}")


# --------------------------------------------------------------------------- #
# distillation-teacher refresh (teacher := current champion)
# --------------------------------------------------------------------------- #
def build_champion_teacher_map(
    store_df: Any, champion_dir: str | Path, *, backend: Any,
    bin_width: float = 0.05,
) -> tuple[dict[str, str], list[str]]:
    """``({cell_key: champion_dir}, per_row_cell_keys)`` — the champion as the
    single teacher for every cell present in the (converged) corpus.

    The per-row cell key is the serve-recipe ``cyclen_cell_key(feed, e_core)`` (the
    same key the per-cell calibration + distillation use), so the refreshed cache
    aligns row-for-row with the training corpus.  Returns the teacher map plus the
    aligned key list for :func:`lpopt.model.distill.build_soft_targets`.
    """
    from .cell_calibrate import cyclen_cell_key
    from ..vendor.masterrl.domain import CaseKey
    from ..data.schema import unpack_pattern

    keys: list[str] = []
    for _, row in store_df.iterrows():
        try:
            pat = unpack_pattern(str(row["pattern"]))
            e_core, _ = backend.cyclen_e_core(pat)
        except Exception:  # noqa: BLE001 — an unresolved row keys to its stored e_core
            e_core = row.get("e_core")
        keys.append(cyclen_cell_key(row.get("feed"), e_core, bin_width))
    teachers = {k: str(champion_dir) for k in sorted(set(keys))}
    return teachers, keys


def refresh_distill_cache(
    champion_dir: str | Path,
    *,
    out_path: str | Path = "data/models/_v5_distill_soft.npz",
    store_dir: str | Path = "data/store",
    library_id: str = "ga80",
    target_names: list[str] | None = None,
    device: str = "cpu",
) -> dict:
    """Rebuild the soft-target distillation cache with the CURRENT champion as the
    teacher for every corpus cell, writing ``out_path`` (shipped with the push).

    Real-launch step (not run under ``--dry-run``): loads the champion backend,
    keys every converged store row to its serve cell, and scores the champion into
    a fresh ``_v5_distill_soft.npz``.  Returns the build summary."""
    from .model_api import PosValCnnBackend
    from .distill import build_soft_targets
    from ..data.store import StoreReader

    df = StoreReader(store_dir).records
    df = df[df["converged"] == True] if "converged" in df.columns else df  # noqa: E712
    backend = PosValCnnBackend.from_dir(
        champion_dir, store_dir=store_dir, library_id=library_id, device=device)
    tnames = target_names or list(getattr(backend, "target_names", []))
    teachers, keys = build_champion_teacher_map(df, champion_dir, backend=backend)
    return build_soft_targets(
        df, teachers, cell_keys=keys, target_names=tnames,
        store_dir=store_dir, device=device, library_id=library_id, out_path=out_path)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan_al_retrain(
    champion_dir: str | Path,
    *,
    input_deck: str = "lpopt.inp",
    distill_cache: str | Path = "data/models/_v5_distill_soft.npz",
    fr_rank_weight: float = 0.1,
    map_fr_consistency_weight: float = 0.0,
    refresh_teacher: bool = True,
    add_fxy_head: bool = False,
    arm4: bool = False,
    arm5: bool = False,
) -> dict:
    """Compose the champion-faithful AL retrain plan (no side effects).

    Returns the recovered recipe, the distillation-teacher refresh step (teacher :=
    ``champion_dir``), the composed ``lpopt.model.train`` args, and the full remote
    invocation.  Callers dry-run by printing; a live run executes the refresh then
    ``push`` + the remote invocation.

    ``add_fxy_head`` composes the F_xy switch's phase-P4 recipe instead of a plain
    retrain: ``--init-from <champion> --freeze-trunk-cyclen --promote-fxy``.  The
    trunk (and cyclen) stay byte-identical to the champion while ONE new head row
    learns f_xy against its fitted F_r prior — so the seven legacy targets cannot
    regress by construction, which is what makes the honest gate cheap to pass
    with ~840 labels.  It is deliberately a plan-level switch, not a recipe field:
    the champion did not have the head, and pretending its recipe did would be the
    same drift the ``cond_schema`` fix closed.

    ``arm4`` (prereg Amendment D.4.2) composes that same fine-tune from the
    PROMOTED champion with three changes and nothing else: the trunk is
    fine-tuned at ``lr * 0.05`` instead of frozen (``--trunk-finetune-lr-mult``,
    which keeps the cyclen row mask), the f_xy row synthesizes against the
    model's OWN predicted F_r (``--fxy-prior-on-predicted``), and selection /
    warmup are pinned to arm 3's ``--fxy-select-weight 0.5 --warmup-epochs 2``.
    It implies ``add_fxy_head`` (the same ``--init-from <champion>
    --promote-fxy`` spine), so the distillation teacher is the champion passed
    in — for arm 4 that MUST be ``data/models/s1j``, since a stale ``s1i`` cache
    would pull the honest gate toward the OLD champion exactly when the trunk is
    unfrozen (D.4.2 [1]).

    ``arm5`` (prereg Amendment E.3.2) is arm 4's command **verbatim** plus the
    within-cell pairwise rank hinge on the composed f_xy row and the elite
    selection band: ``--fxy-rank-weight 3.0 --fxy-rank-cell gate
    --fxy-rank-margin-z 0.1 --fxy-rank-min-gap 0.005 --fxy-rank-low-thresh 1.60
    --fxy-rank-low-weight 3.0 --fxy-select-band 0.50``.  It IMPLIES ``arm4``,
    and it deliberately changes nothing else — including the distillation
    teacher, which stays the champion passed in — because the whole point of the
    arm is a PAIRED comparison against arm 4: if any other knob moved, a ranking
    difference could not be attributed to the objective (E.3)."""
    arm4 = bool(arm4 or arm5)
    recipe = champion_recipe(champion_dir)
    if add_fxy_head or arm4:
        recipe = {**recipe, "promote_fxy": True}
    uses_distill = float(recipe.get("distill_weight", 0.0)) > 0.0
    cache = str(distill_cache) if uses_distill else None
    train_args = recipe_to_train_args(
        recipe, distill_cache=cache, fr_rank_weight=fr_rank_weight,
        map_fr_consistency_weight=map_fr_consistency_weight,
        init_from=(str(champion_dir) if (add_fxy_head or arm4) else None),
        freeze_trunk_cyclen=bool(add_fxy_head and not arm4),
        trunk_finetune_lr_mult=(ARM4_TRUNK_LR_MULT if arm4 else 0.0),
        fxy_prior_on_predicted=bool(arm4),
        fxy_select_weight=(ARM4_FXY_SELECT_WEIGHT if arm4 else None),
        warmup_epochs=(ARM4_WARMUP_EPOCHS if arm4 else None),
        fxy_rank=bool(arm5),
        fxy_select_band=(ARM5_FXY_SELECT_BAND if arm5 else None))
    refresh_cmd = (
        f"python -c \"from lpopt.model.al_retrain import refresh_distill_cache; "
        f"refresh_distill_cache({str(champion_dir)!r}, out_path={str(distill_cache)!r})\""
        if (uses_distill and refresh_teacher) else None)
    return {
        "champion": str(champion_dir),
        "recipe": recipe,
        "distill_teacher_refresh": {
            "enabled": bool(uses_distill and refresh_teacher),
            "teacher": str(champion_dir),
            "cache": str(distill_cache) if uses_distill else None,
            "command": refresh_cmd,
        },
        "train_args": train_args,
        "steps": [s for s in (
            refresh_cmd,
            f"python -m lpopt.remote --input {input_deck} push",
            remote_invocation(train_args, input_deck=input_deck),
        ) if s],
        "gate_after": (
            "python gate_promote.py data/models/<new_ts> --promote   "
            "# honest no-regression gate vs the current champion; promote only on PASS"),
    }


def _print_plan(plan: dict) -> None:
    r = plan["recipe"]
    print(f"champion recipe ({Path(plan['champion']).name}): "
          f"width={r['width']} n_blocks={r['n_blocks']} ensemble={r['n_members']} "
          f"split={r['split']}")
    print(f"  switches: physics_prior={r['cyclen_physics_prior']} "
          f"quantile_heads={r['quantile_heads']}(w={r['quantile_weight']}) "
          f"distill_w={r['distill_weight']} promote_asm_bu={r['promote_max_asm_bu']} "
          f"promote_fxy={r['promote_fxy']} "
          f"cyclen_rank={r['cyclen_rank_weight']} auto_cell_cal={r['auto_fit_cell_calibration']}")
    tr = plan["distill_teacher_refresh"]
    print(f"  distill teacher refresh: enabled={tr['enabled']} teacher={Path(tr['teacher']).name if tr['teacher'] else None} "
          f"cache={tr['cache']}")
    print("\ncomposed retrain steps:")
    for i, s in enumerate(plan["steps"], 1):
        print(f"  [{i}] {s}")
    print(f"  [gate] {plan['gate_after']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.model.al_retrain")
    ap.add_argument("--champion", required=True, help="champion model dir to reproduce")
    ap.add_argument("--input", default="lpopt.inp", help="deck with the [remote] table")
    ap.add_argument("--distill-cache", default="data/models/_v5_distill_soft.npz")
    ap.add_argument("--f-r-rank-weight", type=float, default=0.1)
    ap.add_argument("--map-fr-consistency-weight", type=float, default=0.0)
    ap.add_argument("--no-refresh-teacher", dest="refresh_teacher",
                    action="store_false", default=True)
    ap.add_argument("--add-fxy-head", action="store_true",
                    help="compose the F_xy phase-P4 freeze-finetune recipe: "
                         "--init-from <champion> --freeze-trunk-cyclen "
                         "--promote-fxy (frozen trunk, one new head row)")
    ap.add_argument("--arm4", action="store_true",
                    help="compose prereg Amendment D.4.2's arm-4 command "
                         "(implies --add-fxy-head): the champion trunk is "
                         f"fine-tuned at lr x {ARM4_TRUNK_LR_MULT} instead of "
                         "frozen, --fxy-prior-on-predicted replaces the direct "
                         f"f_xy row, --fxy-select-weight "
                         f"{ARM4_FXY_SELECT_WEIGHT} --warmup-epochs "
                         f"{ARM4_WARMUP_EPOCHS}; pass --champion data/models/s1j "
                         "(also the distill teacher)")
    ap.add_argument("--arm5", action="store_true",
                    help="compose prereg Amendment E.3.2's arm-5 command "
                         "(implies --arm4, whose flags are carried verbatim): "
                         f"adds --fxy-rank-weight {ARM5_FXY_RANK_WEIGHT} "
                         f"--fxy-rank-cell {ARM5_FXY_RANK_CELL} "
                         f"--fxy-rank-margin-z {ARM5_FXY_RANK_MARGIN_Z} "
                         f"--fxy-rank-min-gap {ARM5_FXY_RANK_MIN_GAP} "
                         f"--fxy-rank-low-thresh {ARM5_FXY_RANK_LOW_THRESH} "
                         f"--fxy-rank-low-weight {ARM5_FXY_RANK_LOW_WEIGHT} "
                         f"--fxy-select-band {ARM5_FXY_SELECT_BAND}; the "
                         "teacher cache is UNCHANGED (paired comparison)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the composed invocation only (no refresh, no train)")
    ap.add_argument("--execute-refresh", action="store_true",
                    help="build the refreshed distill cache now (still no training)")
    args = ap.parse_args(argv)

    plan = plan_al_retrain(
        args.champion, input_deck=args.input, distill_cache=args.distill_cache,
        fr_rank_weight=args.f_r_rank_weight,
        map_fr_consistency_weight=args.map_fr_consistency_weight,
        refresh_teacher=args.refresh_teacher,
        add_fxy_head=args.add_fxy_head, arm4=args.arm4, arm5=args.arm5)
    _print_plan(plan)

    if args.execute_refresh and plan["distill_teacher_refresh"]["enabled"]:
        print("\n[execute] refreshing distill cache (teacher := champion)...")
        summary = refresh_distill_cache(args.champion, out_path=args.distill_cache)
        print(f"[execute] wrote {args.distill_cache}: "
              f"{summary.get('n_scored', '?')} rows scored")
    elif not args.dry_run:
        print("\n(nothing executed; pass --dry-run to acknowledge, or --execute-refresh "
              "to build the teacher cache; retrain #3 launches on boundary round-1 data)")
    return 0


__all__ = [
    "ARM4_TRUNK_LR_MULT", "ARM4_FXY_SELECT_WEIGHT", "ARM4_WARMUP_EPOCHS",
    "ARM5_FXY_RANK_WEIGHT", "ARM5_FXY_RANK_CELL", "ARM5_FXY_RANK_MARGIN_Z",
    "ARM5_FXY_RANK_MIN_GAP", "ARM5_FXY_RANK_LOW_THRESH",
    "ARM5_FXY_RANK_LOW_WEIGHT", "ARM5_FXY_SELECT_BAND",
    "champion_recipe", "recipe_to_train_args", "remote_invocation",
    "build_champion_teacher_map", "refresh_distill_cache", "plan_al_retrain",
]


if __name__ == "__main__":
    raise SystemExit(main())
