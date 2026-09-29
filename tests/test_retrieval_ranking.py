import numpy as np

from helpers import *  # noqa: F401,F403  (sets sys.path)
from agentrec.ranking.ranker import Ranker
from agentrec.retrieval.index import ExactIndex, IVFIndex
from agentrec.retrieval.retriever import CandidateRetriever


def _vecs(n=300, d=8, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, d))
    return V / np.linalg.norm(V, axis=1, keepdims=True), rng.normal(size=(5, d))


def test_exact_index_matches_bruteforce():
    V, Q = _vecs()
    idx, _ = ExactIndex(V).search(Q, 10)
    assert np.array_equal(idx, np.argsort(-(Q @ V.T), axis=1)[:, :10])


def test_ivf_with_all_probes_is_exact():
    V, Q = _vecs()
    ivf = IVFIndex(V, n_lists=8, n_probe=8)
    a, _ = ivf.search(Q, 10)
    b, _ = ExactIndex(V).search(Q, 10)
    assert np.array_equal(a, b)


def test_ivf_fewer_probes_is_approximate_but_reasonable():
    V, Q = _vecs(n=2000)
    a, _ = IVFIndex(V, n_lists=32, n_probe=4).search(Q, 20)
    b, _ = ExactIndex(V).search(Q, 20)
    recall = np.mean([len(set(x) & set(y)) / 20 for x, y in zip(a, b)])
    assert 0.3 < recall <= 1.0


def test_exclusion_is_respected():
    V, Q = _vecs()
    ex = set(np.argsort(-(Q[0] @ V.T))[:5].tolist())
    idx, _ = ExactIndex(V).search(Q[:1], 10, exclude=[ex])
    assert not ex & set(idx[0].tolist())


def test_retriever_cold_user_uses_popularity_and_mask():
    V, _ = _vecs()
    pop = np.arange(300)
    r = CandidateRetriever(ExactIndex(V), V, pop)
    c = r.retrieve(None, 10)
    assert c.sources == ["popular"] * 10 and c.items.tolist() == list(range(10))
    allowed = np.zeros(300, bool); allowed[100:] = True
    c2 = r.retrieve(V[0], 20, allowed=allowed)
    assert (c2.items >= 100).all() and len(set(c2.items.tolist())) == 20


def test_rankers_learn_a_separable_signal():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(60):
        F = rng.normal(size=(30, 9))
        y = (F[:, 2] + 0.3 * rng.normal(size=30) > 1.0).astype(int)
        groups.append((F, y))
    for kind in ("pointwise", "pairwise", "gbdt"):
        r = Ranker(kind).fit(groups)
        F = rng.normal(size=(200, 9))
        s = r.score(F)
        rank_corr = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(F[:, 2])))[0, 1]
        assert rank_corr > 0.6, kind


def test_identity_ranker_keeps_order():
    r = Ranker("identity").fit([])
    assert np.array_equal(np.argsort(-r.score(np.zeros((5, 9)))), np.arange(5))


def test_exact_index_never_returns_excluded_when_too_few_allowed():
    V, Q = _vecs(n=20)
    ex = set(range(17))                        # only 3 items allowed
    idx, _ = ExactIndex(V).search(Q[:1], 10, exclude=[ex])
    valid = [i for i in idx[0] if i >= 0]
    assert sorted(valid) == [17, 18, 19]
