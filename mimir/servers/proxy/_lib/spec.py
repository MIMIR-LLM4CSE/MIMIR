"""What a session optimises and under what: objectives (the first the primary), constraints
(a threshold, ``of_baseline`` a factor of the baseline's value, or both: the looser), directives
the agent keeps to, the program's own metrics (``declare``) and the hardware's bound on the
primary objective (``bound``, with the calculation it comes from). The config holds it as it
always did (``primary_metric``, ``primary_goal``, ``requirements``) plus ``secondary``,
``directives``, ``declared``, ``bound`` and ``bound_basis``.
"""

from __future__ import annotations

from . import metric_catalog
from .metrics import VALID_OPT_OPERATORS

def of(cfg: dict) -> dict:
    """The spec *cfg* holds."""
    primary = {"metric": cfg.get("primary_metric", "time_s"), "goal": cfg.get("primary_goal", "min")}
    return {
        "objectives": [primary, *[dict(o) for o in cfg.get("secondary") or []]],
        "constraints": [dict(r) for r in cfg.get("requirements") or []],
        "directives": list(cfg.get("directives") or []),
        "declare": [dict(d) for d in cfg.get("declared") or []],
        **({"bound": cfg["bound"], "bound_basis": cfg.get("bound_basis", "")} if cfg.get("bound") else {}),
    }


def merged(base: dict, given: dict | None) -> dict:
    """*base* with what *given* sets: a key given replaces the base's, and ``change`` patches
    by metric the objective or constraint naming it (``{"metric", "tolerance"?, "goal"?,
    "threshold"?, "of_baseline"?, "operator"?, "remove"?}``). ``reason`` is not part of it."""
    out = {key: value for key, value in base.items()}
    for key, value in (given or {}).items():
        if value is not None and key not in ("change", "reason"):
            out[key] = value
    for patch in (given or {}).get("change") or []:
        name = patch.get("metric")
        fields = {k: v for k, v in patch.items() if k not in ("metric", "remove")}
        for key in ("objectives", "constraints"):
            rows = [dict(r) for r in out.get(key) or []]
            hit = [r for r in rows if r.get("metric") == name
                   and (key == "objectives" or not patch.get("operator") or r.get("operator") == patch["operator"])]
            if patch.get("remove"):
                rows = [r for r in rows if r not in hit]
            for row in hit:
                for k, v in fields.items():
                    if key == "constraints" or k in ("goal", "tolerance"):
                        # A null drops the field: the bound kept is the other one.
                        row.pop(k, None) if v is None else row.__setitem__(k, v)
            out[key] = rows
    return out


_LOOSER = {"lt": 1, "lte": 1, "gt": -1, "gte": -1}


def loosened(before: dict, after: dict, cfg: dict) -> list[dict]:
    """What *after* lets through that *before* did not: a constraint gone or widened, an
    objective gone or more tolerant; each says whether it concerns the solution's correctness."""
    out = []

    def note(metric: str, what: str) -> None:
        found = metric_catalog.lookup(cfg, metric)
        out.append({"metric": metric, "what": what, "correctness": bool(found and found.source == "numerics")})

    then = {(c["metric"], c.get("operator")): c for c in before.get("constraints") or []}
    now = {(c["metric"], c.get("operator")): c for c in after.get("constraints") or []}
    for key, old in then.items():
        new, sign = now.get(key), _LOOSER.get(key[1], 0)
        if new is None:
            note(key[0], "constraint removed")
            continue
        # Both bounds given, the looser holds: a bound added or widened lets more through.
        for bound in ("threshold", "of_baseline"):
            was, is_ = old.get(bound), new.get(bound)
            if isinstance(is_, (int, float)) and (was is None or sign * (is_ - was) > 0):
                note(key[0], f"{key[1]} {bound} {was} -> {is_}")
    goals = {o["metric"]: o for o in after.get("objectives") or []}
    for old in before.get("objectives") or []:
        new = goals.get(old["metric"])
        if new is None:
            note(old["metric"], "objective removed")
        elif (new.get("tolerance") or 0) > (old.get("tolerance") or 0):
            note(old["metric"], f"tolerance {old.get('tolerance') or 0} -> {new['tolerance']}")
    return out


def check(spec: dict, cfg: dict | None = None, suite: dict | None = None, entry: dict | None = None) -> str | None:
    """Why *spec* cannot drive a session, or None."""
    cfg = {**(cfg or {}), "declared": _declared(spec, cfg or {})}
    objectives = spec.get("objectives") or []
    if not objectives:
        return "A spec needs at least one objective (time_s when nothing else is asked)."
    seen = set()
    for i, objective in enumerate(objectives):
        name = objective.get("metric", "")
        metric = metric_catalog.lookup(cfg, name)
        if metric is None:
            return (f"objectives[{i}]: no metric is named '{name}'. proxy_eval_status(op='metrics') lists "
                    "them; a metric of the program's own is declared first (spec.declare).")
        if name in seen:
            return f"objectives[{i}]: {name} is already an objective."
        seen.add(name)
        if objective.get("goal", metric.goal) not in ("min", "max"):
            return f"objectives[{i}]: goal must be 'min' or 'max'."
        tolerance = objective.get("tolerance")
        if tolerance is not None and (not isinstance(tolerance, (int, float)) or tolerance < 0):
            return f"objectives[{i}]: tolerance is a relative degradation, >= 0 (0.1 is 10%)."
        why = metric_catalog.missing(cfg, metric, suite, entry)
        if why:
            return f"objectives[{i}]: {name} {why}."
    for i, constraint in enumerate(spec.get("constraints") or []):
        if not constraint.get("metric"):
            return f"constraints[{i}] is missing 'metric'."
        if constraint.get("operator") not in VALID_OPT_OPERATORS:
            return f"constraints[{i}]: operator is one of {', '.join(sorted(VALID_OPT_OPERATORS))}."
        relative = constraint.get("of_baseline")
        if constraint.get("threshold") is None and relative is None:
            return (f"constraints[{i}] takes a threshold, of_baseline (a factor of the baseline's value), "
                    "or both (the looser holds).")
        if relative is not None and (not isinstance(relative, (int, float)) or relative <= 0):
            return f"constraints[{i}]: of_baseline is a positive factor (1.1 is 10% above the baseline)."
    bound = spec.get("bound")
    if bound:  # 0 takes the bound away
        if not isinstance(bound, (int, float)) or bound <= 0:
            return "bound is the best the hardware allows for the primary objective, > 0, in its unit."
        if not str(spec.get("bound_basis") or "").strip():
            return ("bound needs a bound_basis: the calculation it comes from, e.g. "
                    "'3 fields x 4 B x 512^3 per step / 2000 GB/s'.")
    if not all(isinstance(d, str) and d.strip() for d in spec.get("directives") or []):
        return "directives are sentences the agent keeps to."
    return None


def apply(cfg: dict, spec: dict) -> dict:
    """*cfg* holding *spec*."""
    primary, *secondary = spec["objectives"]
    known = {**cfg, "declared": _declared(spec, cfg)}
    goal_of = lambda o: o.get("goal") or metric_catalog.lookup(known, o["metric"]).goal  # noqa: E731
    constraints = [dict(c) for c in spec.get("constraints") or []]
    cfg.update({
        "primary_metric": primary["metric"],
        "primary_goal": goal_of(primary),
        "secondary": [{"metric": o["metric"], "goal": goal_of(o),
                       **({"tolerance": float(o["tolerance"])} if o.get("tolerance") is not None else {})}
                      for o in secondary],
        "requirements": constraints,
        "directives": [d.strip() for d in spec.get("directives") or []],
        "declared": known["declared"],
    })
    if spec.get("bound"):
        cfg["bound"], cfg["bound_basis"] = float(spec["bound"]), str(spec["bound_basis"]).strip()
    else:
        cfg.pop("bound", None)
        cfg.pop("bound_basis", None)
    return cfg


def _declared(spec: dict, cfg: dict) -> list[dict]:
    """The program's metrics, as the spec declares them, else as the config holds them."""
    return [dict(d) for d in (spec.get("declare") if spec.get("declare") is not None
                              else cfg.get("declared") or [])]


def absolute(requirements: list[dict]) -> list[dict]:
    """The constraints a run checks by itself: those with a threshold of their own."""
    return [r for r in requirements if r.get("of_baseline") is None]


def relative(requirements: list[dict]) -> list[dict]:
    """The constraints read against the baseline's value: settled with the verdict."""
    return [r for r in requirements if r.get("of_baseline") is not None]
