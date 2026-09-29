import math

import pytest

from helpers import *  # noqa: F401,F403  (sets sys.path)
from agentrec.evaluation.metrics import (average_precision_at_k, evaluate_lists, hit_rate_at_k, mrr_at_k,
                                         ndcg_at_k, precision_at_k, recall_at_k)

RANKED = [3, 1, 7, 5]
REL = {1, 5, 9}


def test_hand_computed_values():
    # hits at positions 2 and 4
    assert precision_at_k(RANKED, REL, 4) == pytest.approx(0.5)
    assert recall_at_k(RANKED, REL, 4) == pytest.approx(2 / 3)
    assert hit_rate_at_k(RANKED, REL, 4) == 1.0
    assert mrr_at_k(RANKED, REL, 4) == pytest.approx(0.5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 4) == pytest.approx(dcg / idcg)
    assert average_precision_at_k(RANKED, REL, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_perfect_and_empty_lists():
    assert ndcg_at_k([1, 5, 9], REL, 3) == pytest.approx(1.0)
    assert average_precision_at_k([1, 5, 9], REL, 3) == pytest.approx(1.0)
    assert ndcg_at_k([2, 4, 6], REL, 3) == 0.0
    assert mrr_at_k([2, 4, 6], REL, 3) == 0.0


def test_k_truncates():
    assert precision_at_k(RANKED, REL, 1) == 0.0
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(1 / 3)


def test_recall_undefined_for_empty_target():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 4)


def test_evaluate_lists_averages_over_users_with_targets():
    out = evaluate_lists({0: [1, 2], 1: [3, 4], 2: [5]}, {0: {1}, 1: {9}, 2: set()}, ks=[2])
    assert out["n_users"] == 2
    assert out["hit@2"] == pytest.approx(0.5)
