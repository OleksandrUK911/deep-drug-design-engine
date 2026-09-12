import numpy as np

from ml.rl.pareto import non_dominated_front, pareto_knee_points, scalarization_reachable_set


def test_non_dominated_front_simple_case():
    # Point 0 dominates point 2 (better on both axes); point 1 is
    # non-comparable to point 0 (better on axis 0, worse on axis 1).
    values = np.array([
        [1.0, 1.0],  # 0: non-dominated
        [2.0, 0.5],  # 1: non-dominated (better on axis 0)
        [0.5, 0.5],  # 2: dominated by 0
    ])
    front = set(non_dominated_front(values).tolist())
    assert front == {0, 1}


def test_non_dominated_front_all_equal_are_all_nondominated():
    values = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    front = set(non_dominated_front(values).tolist())
    assert front == {0, 1, 2}


def test_non_dominated_front_single_dominant_point():
    values = np.array([[1.0, 1.0], [0.5, 0.5], [0.2, 0.9]])
    front = set(non_dominated_front(values).tolist())
    assert front == {0}


def test_scalarization_reachable_set_finds_extremes_on_convex_front():
    # A simple convex front: extremes should always be reachable by
    # weight vectors that put all weight on one axis.
    values = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ])
    reachable = scalarization_reachable_set(values, n_grid_steps=10)
    assert 0 in reachable  # weight (1,0) selects point 0
    assert 2 in reachable  # weight (0,1) selects point 2


def test_scalarization_reachable_set_misses_concave_front_point():
    """The classic textbook case: a point on a concave (non-convex)
    region of the Pareto front is never the argmax of ANY linear
    scalarization, even though it's fully non-dominated. This is the
    exact phenomenon ml/training/analyze_pareto.py demonstrates on real
    generated molecules."""
    values = np.array([
        [1.0, 0.0],   # extreme A
        [0.4, 0.4],   # concave "dip" point (sum < 1, inside the A-B segment) -- Pareto-optimal but unreachable
        [0.0, 1.0],   # extreme B
    ])
    front = set(non_dominated_front(values).tolist())
    assert front == {0, 1, 2}  # all three are non-dominated

    reachable = scalarization_reachable_set(values, n_grid_steps=20)
    assert 1 not in reachable  # the concave point is never a linear-scalarization argmax
    assert 0 in reachable and 2 in reachable


def test_pareto_knee_points_picks_balanced_candidate():
    values = np.array([
        [1.0, 0.0],   # extreme, unbalanced
        [0.5, 0.5],   # balanced -- should be the knee
        [0.0, 1.0],   # extreme, unbalanced
    ])
    front_idx = non_dominated_front(values)
    knees = pareto_knee_points(values, front_idx, n_knees=1)
    assert knees[0] == 1
