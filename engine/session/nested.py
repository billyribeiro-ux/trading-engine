"""
Nested multi-scale structure.

A single scale either collapses the afternoon into one 5-hour leg (too coarse) or
explodes it into noise (too fine). The professional view is HIERARCHICAL: the
major skeleton (the 4-leg story of the day) with each major leg's internal
sub-structure nested beneath it, drawn from a finer scale that lives strictly
inside the major leg's time span.

This is the "capture every ebb and flow" requirement made digestible: you see the
day's spine first, then expand any major move into its component swings.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pivots import Leg, MultiScaleDecomposition


@dataclass(frozen=True)
class NestedLeg:
    """A major leg with the finer-scale legs nested inside its time span."""

    major: Leg
    sub_legs: tuple[Leg, ...]  # finer legs within [start_index, end_index]
    sub_scale: float  # the scale the sub-legs came from

    @property
    def n_sub(self) -> int:
        return len(self.sub_legs)


def _legs_within(legs: list[Leg], start_idx: int, end_idx: int) -> list[Leg]:
    """Finer legs whose span falls inside the major leg [start_idx, end_idx]."""
    out = []
    for lg in legs:
        if lg.start_index >= start_idx and lg.end_index <= end_idx:
            out.append(lg)
    return out


def build_nested_structure(
    decomposition: MultiScaleDecomposition,
    major_scale: float,
    sub_scale: float | None = None,
    merge: bool = True,
) -> list[NestedLeg]:
    """
    Nest finer legs inside each major leg.

    major_scale : the scale defining the day's skeleton.
    sub_scale   : the finer scale to nest inside each major leg. If None, picks
                  the next-finer available scale that yields sub-structure.
    merge       : apply the swing-significance merge to the major legs so the
                  skeleton matches the dissection's classified legs (one clean
                  flush, etc.). MUST be True to keep the STRUCTURE display
                  consistent with the LEG ROLES section -- they are the same
                  legs viewed two ways.
    """
    from .pivots import build_skeleton

    major_legs = build_skeleton(decomposition, major_scale)
    if not major_legs:
        return []

    finer_scales = sorted((s for s in decomposition.scales if s < major_scale), reverse=True)
    if sub_scale is None:
        sub_scale = finer_scales[0] if finer_scales else major_scale
    sub_legs_all = decomposition.legs_by_scale.get(sub_scale, [])

    nested: list[NestedLeg] = []
    for mleg in major_legs:
        subs = _legs_within(sub_legs_all, mleg.start_index, mleg.end_index)
        if len(subs) <= 1:
            subs = []
        nested.append(NestedLeg(major=mleg, sub_legs=tuple(subs), sub_scale=sub_scale))
    return nested
