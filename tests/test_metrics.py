"""Hand-built task matrices for the metric formulas."""

from __future__ import annotations

import pytest

from valmorph.metrics import (
    average_accuracy,
    backward_transfer,
    forgetting,
    learning_accuracy,
)


def test_three_tasks_match_the_written_formulas() -> None:
    rows = [
        [0.5, 0.0, 0.0],
        [0.4, 0.8, 0.0],
        [0.2, 0.7, 0.9],
    ]
    assert average_accuracy(rows) == pytest.approx((0.2 + 0.7 + 0.9) / 3)
    assert learning_accuracy(rows) == pytest.approx((0.5 + 0.8 + 0.9) / 3)
    assert backward_transfer(rows) == pytest.approx((-0.3 + -0.1) / 2)
    assert forgetting(rows) == pytest.approx((0.3 + 0.1) / 2)


def test_one_task_has_no_transfer() -> None:
    rows = [[0.4]]
    assert average_accuracy(rows) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="backward transfer"):
        backward_transfer(rows)
    with pytest.raises(ValueError, match="forgetting"):
        forgetting(rows)


def test_a_ragged_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="square"):
        average_accuracy([[0.1, 0.2], [0.3]])
