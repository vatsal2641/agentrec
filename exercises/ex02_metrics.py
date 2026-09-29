"""Exercise 02: ranking metrics. Reference: src/agentrec/evaluation/metrics.py (don't peek)."""
import math


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    raise NotImplementedError


def mrr_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    raise NotImplementedError


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """binary relevance, DCG = sum rel_k / log2(k+1); IDCG uses min(k, |relevant|) ones."""
    raise NotImplementedError


def ap_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """(1/min(k,|R|)) * sum_k P@k * rel_k"""
    raise NotImplementedError


if __name__ == "__main__":
    r, rel = [3, 1, 7, 5], {1, 5, 9}
    assert abs(recall_at_k(r, rel, 4) - 2 / 3) < 1e-9
    assert mrr_at_k(r, rel, 4) == 0.5 and mrr_at_k(r, rel, 1) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert abs(ndcg_at_k(r, rel, 4) - dcg / (1 + 1 / math.log2(3) + 0.5)) < 1e-9
    assert abs(ap_at_k(r, rel, 4) - 1 / 3) < 1e-9
    assert ndcg_at_k([1, 5, 9], rel, 3) == 1.0
    print("all tests passed. Now: why is NDCG@4 not 1 even though both hits are in the top 4?")
