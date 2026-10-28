"""Accuracy summaries for a task-by-task matrix.

``rows[i][j]`` is accuracy on task ``j`` after task ``i`` finishes. Both
indexes are zero-based. The formulas are the ones in ``docs/05_METRICS.md``.
Backward transfer is the same expression as Lopez-Paz and Ranzato, equation 3.
"""

from __future__ import annotations


def _square(rows: list[list[float]]) -> int:
    count = len(rows)
    if count < 1 or any(len(row) != count for row in rows):
        raise ValueError("rows must be a non-empty square matrix")
    return count


def average_accuracy(rows: list[list[float]]) -> float:
    """Mean of the last row. Ledger name ``average_accuracy``."""

    count = _square(rows)
    return sum(rows[-1]) / count


def learning_accuracy(rows: list[list[float]]) -> float:
    """Mean of the end-of-task diagonal. Ledger name ``learning_accuracy``."""

    count = _square(rows)
    return sum(rows[task][task] for task in range(count)) / count


def backward_transfer(rows: list[list[float]]) -> float:
    """Mean of final minus end-of-task accuracy, over the earlier tasks.

    Ledger name ``bwt_end_minus_diag``. One task has no earlier task.
    """

    count = _square(rows)
    if count < 2:
        raise ValueError("backward transfer needs at least two tasks")
    final = rows[-1]
    change = sum(final[task] - rows[task][task] for task in range(count - 1))
    return change / (count - 1)


def forgetting(rows: list[list[float]]) -> float:
    """Mean, over earlier tasks, of the peak before the end minus the final accuracy.

    For task ``j`` the peak is the maximum of column ``j`` from that task's own
    row through the second-to-last row. Ledger name ``forgetting``. Higher
    means more forgotten. One task has no earlier task.
    """

    count = _square(rows)
    if count < 2:
        raise ValueError("forgetting needs at least two tasks")
    total = 0.0
    for task in range(count - 1):
        peak = max(rows[seen][task] for seen in range(task, count - 1))
        total += peak - rows[-1][task]
    return total / (count - 1)
