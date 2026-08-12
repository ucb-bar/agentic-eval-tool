"""Arms, seed pairing, and the fairness checks that make a cross-arm comparison mean something.

``aet`` could already run N methods × M seeds and compare them. What it could not do is state, and
then verify, that two conditions differ in exactly one declared way. "Arm" existed only as a plot
label, so the claim that arms were otherwise identical rested on whoever wrote the driver script
having got it right — which is not a control, because nobody can re-derive it afterwards.

This package makes the treatment explicit (:class:`Arm`), the grid explicit (:class:`ArmMatrix`),
and the fairness claim mechanically checkable (:meth:`ArmMatrix.validate`). Paired analysis over the
resulting cells lives in :mod:`aet.experiment.paired`.
"""

from aet.experiment.arm import Arm, ArmDiff, diff_arms
from aet.experiment.matrix import ArmMatrix, Cell
from aet.experiment.paired import ArmComparison, compare_arms, pair_by_seed

__all__ = [
    "Arm",
    "ArmDiff",
    "diff_arms",
    "ArmMatrix",
    "Cell",
    "ArmComparison",
    "compare_arms",
    "pair_by_seed",
]
