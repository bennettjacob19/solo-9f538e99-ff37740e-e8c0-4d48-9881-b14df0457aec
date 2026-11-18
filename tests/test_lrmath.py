"""Determinism and math-equivalence unit tests for the shared LR library."""

import numpy as np

from app import lrmath as lr


def _toy_dataset(n=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    # Label generated from a fixed linear rule so both classes are present.
    z = 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.2
    y = (lr.sigmoid(z) > 0.5).astype(np.float64)
    return x, y


def test_zero_initialization_and_shapes():
    z = lr.zero_buffers()
    assert z.shape == (3,) and np.all(z == 0)
    x, _ = _toy_dataset()
    assert lr.augment(x).shape == (len(x), 3)


def test_concatenated_residuals_rebuild_exact_full_gradient():
    x, y = _toy_dataset()
    x_aug = lr.augment(x)
    w = np.array([0.1, -0.2, 0.3])
    g_full, _ = lr.shard_gradient(x_aug, y, w)

    cut = (len(y) + 1) // 2
    r1, _ = lr.shard_residuals(x_aug[:cut], y[:cut], w)
    r2, _ = lr.shard_residuals(x_aug[cut:], y[cut:], w)
    g_rebuilt = lr.full_average_gradient(
        x_aug, np.concatenate([r1, r2]))
    # Same single matrix product and summation order as the reference trainer.
    np.testing.assert_array_equal(g_rebuilt, g_full)


def test_distributed_two_shard_matches_single_process_reference():
    x, y = _toy_dataset()
    x_aug = lr.augment(x)
    cut = (len(y) + 1) // 2
    xa, ya = x_aug[:cut], y[:cut]
    xb, yb = x_aug[cut:], y[cut:]

    w = lr.zero_buffers()
    v = lr.zero_buffers()
    ws, vs, losses_ref = lr.reference_train(x, y, 30)
    dist_w, dist_v, dist_loss = [w.copy()], [v.copy()], [lr.bce_loss(x_aug, y, w)]
    for _ in range(30):
        ra, _ = lr.shard_residuals(xa, ya, w)
        rb, _ = lr.shard_residuals(xb, yb, w)
        g = lr.full_average_gradient(x_aug, np.concatenate([ra, rb]))
        w, v = lr.momentum_update(w, v, g)
        dist_w.append(w.copy())
        dist_v.append(v.copy())
        dist_loss.append(lr.bce_loss(x_aug, y, w))

    for k in range(31):
        np.testing.assert_array_equal(dist_w[k], ws[k])
        np.testing.assert_array_equal(dist_v[k], vs[k])
        assert dist_loss[k] == losses_ref[k]


def test_loss_decreases_on_separable_toy_data():
    x, y = _toy_dataset()
    _, _, losses = lr.reference_train(x, y, 50)
    assert losses[-1] < losses[0]
    assert all(np.isfinite(losses))


def test_bit_identical_repeat_runs():
    x, y = _toy_dataset()
    run1 = lr.reference_train(x, y, 25)
    run2 = lr.reference_train(x, y, 25)
    for a, b in zip(run1[0], run2[0]):
        np.testing.assert_array_equal(a, b)


def test_sigmoid_stable_at_extremes():
    p = lr.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert p[0] == 0.0 and p[1] == 0.5 and p[2] == 1.0
