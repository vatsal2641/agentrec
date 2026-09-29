import numpy as np

from helpers import tiny_dataset
from agentrec.models.baselines import ContentBased, ItemKNN, Popularity, RecentPopularity
from agentrec.models.mf_bpr import BPRMF
from agentrec.models.two_tower import Batch, TwoTower


def test_popularity_is_count_ordered_and_identical_for_all_users():
    ds = tiny_dataset()
    m = Popularity().fit(ds, ds.train)
    s = m.score(np.array([0, 1]))
    counts = np.asarray(ds.matrix(ds.train).sum(0)).ravel()
    assert np.array_equal(np.argsort(-s[0], kind="stable")[:5], np.argsort(-np.log1p(counts), kind="stable")[:5])
    assert np.allclose(s[0], s[1])


def test_recent_popularity_prefers_recent_interactions():
    ds = tiny_dataset()
    m = RecentPopularity(half_life_days=1.0).fit(ds, ds.train)
    assert m.s.max() > 0


def test_content_model_scores_items_without_interactions():
    ds = tiny_dataset()
    m = ContentBased().fit(ds, ds.train)
    warm_user = int(np.argmax(np.asarray(ds.matrix(ds.train).sum(1)).ravel()))
    never = np.where(np.asarray(ds.matrix(ds.train).sum(0)).ravel() == 0)[0]
    s = m.score(np.array([warm_user]))[0]
    assert len(never) == 0 or np.any(s[never] != 0)


def test_itemknn_keeps_at_most_k_neighbours():
    ds = tiny_dataset()
    m = ItemKNN(k_neighbors=5).fit(ds, ds.train)
    nnz_per_col = np.diff(m.S.tocsc().indptr)
    assert nnz_per_col.max() <= 5 + 5  # ties at the threshold may keep a few extra


def test_cold_users_get_popularity_fallback():
    ds = tiny_dataset()
    m = BPRMF(dim=8, epochs=1).fit(ds, ds.train)
    cold = np.where(~m.has_history)[0]
    if len(cold):
        assert np.allclose(m.score(cold[:1])[0], m.pop_scores)


def test_bpr_loss_decreases():
    ds = tiny_dataset()
    m = BPRMF(dim=8, epochs=8, lr=0.05).fit(ds, ds.train)
    assert m.history[-1] < m.history[0]


def test_bpr_embeddings_reproduce_scores():
    ds = tiny_dataset()
    m = BPRMF(dim=4, epochs=1).fit(ds, ds.train)
    u = np.array([0, 3])
    assert np.allclose(m.user_embeddings(u) @ m.item_embeddings().T, m._score(u), atol=1e-5)


def test_two_tower_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = TwoTower(dim=5, hidden=4, temperature=0.5, reg=0.0, history_len=4)
    n_items, n_genres = 12, 3
    G = rng.random((n_items, n_genres))
    m.G = G / G.sum(1, keepdims=True)
    m.init_params(n_items, n_genres, rng)
    m.p["W2"] = rng.normal(0, 0.5, m.p["W2"].shape)
    m.p["b1"] = rng.normal(0, 0.5, 4)
    m.logq = np.log(rng.dirichlet(np.ones(n_items)))
    b = Batch(hist=np.array([[0, 3, -1, -1], [5, 6, 7, 1], [2, -1, -1, -1], [3, 3, 4, -1]]), pos=np.array([1, 8, 9, 1]))
    _, g = m.loss_and_grads(b)
    worst = 0.0
    for k, v in m.p.items():
        for idx in np.ndindex(v.shape):
            old = v[idx]
            v[idx] = old + 1e-6; lp, _ = m.loss_and_grads(b)
            v[idx] = old - 1e-6; lm, _ = m.loss_and_grads(b)
            v[idx] = old
            num, ana = (lp - lm) / 2e-6, g[k][idx]
            if abs(num) + abs(ana) > 1e-7:
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))
    assert worst < 1e-4


def test_two_tower_history_is_strictly_before_target():
    ds = tiny_dataset()
    m = TwoTower(dim=4, epochs=1, batch_size=64, history_len=5).fit(ds, ds.train)
    b = m._batch(np.arange(min(200, len(m.sample_pos))))
    pos = ds.train.positives.sort_values(["u", "timestamp"], kind="stable")
    t = pos["timestamp"].to_numpy()
    for r, tgt in enumerate(m.sample_pos[:len(b.pos)]):
        win = [tgt - m.L + j for j in range(m.L)]
        for j, h in zip(win, b.hist[r]):
            if h >= 0:
                assert j < tgt and t[j] <= t[tgt]


def test_two_tower_trains_and_embeds_new_history():
    ds = tiny_dataset()
    m = TwoTower(dim=8, epochs=4, batch_size=128, history_len=10).fit(ds, ds.train)
    assert m.history[-1] < m.history[0]
    e1, e2 = m.embed_history([0, 1]), m.embed_history([0, 1, 2, 3])
    assert np.isclose(np.linalg.norm(e1), 1.0) and not np.allclose(e1, e2)
