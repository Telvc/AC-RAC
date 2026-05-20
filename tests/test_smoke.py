from pathlib import Path

import numpy as np
import torch

from rac.acrac import _select_candidate, calibrate_lambdas_from_candidates, predict_acrac
from rac.conformal import precompute_candidates, select_action_max_min
from rac.data import COVID_UTILITY_MATRIX, MOVIELENS_UTILITY_MATRIX, make_utility_fn, save_prediction_cache
from rac.metrics import run_acrac, run_all_methods
from experiments.movielens.run_3_seeds import (
    aggregate_seed_results,
    get_or_create_predictions_for_seed,
    parse_float_list,
    parse_seed_list,
)


def test_cached_covid_one_alpha_smoke():
    root = Path("data/cached/covid")
    cal_probs = np.load(root / "cal_probs.npy")[:64]
    cal_labels = np.load(root / "cal_labels.npy")[:64]
    test_probs = np.load(root / "test_probs.npy")[:64]
    test_labels = np.load(root / "test_labels.npy")[:64]
    results = run_all_methods(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        0.05,
        [0, 1, 2, 3],
        make_utility_fn(COVID_UTILITY_MATRIX),
        ac_kwargs={"max_iter": 5},
    )
    assert set(results) == {"AC-RAC", "RAC", "Score-1", "Score-2"}
    assert all("coverage_overall" in value for value in results.values())


def test_run_all_methods_acrac_smoke():
    rng = np.random.default_rng(11)
    cal_probs = rng.dirichlet(np.ones(4), size=16)
    cal_labels = rng.integers(0, 4, size=16)
    test_probs = rng.dirichlet(np.ones(4), size=8)
    test_labels = rng.integers(0, 4, size=8)
    results = run_all_methods(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        0.05,
        [0, 1, 2, 3],
        make_utility_fn(COVID_UTILITY_MATRIX),
        ac_kwargs={"max_iter": 2, "eta0": 1.0},
    )
    assert set(results) == {"AC-RAC", "RAC", "Score-1", "Score-2"}
    assert len(results["AC-RAC"]["sets"]) == len(test_labels)


def test_movielens_seed_sweep_helpers():
    assert parse_seed_list("0,1,2") == [0, 1, 2]
    assert parse_seed_list("0:3") == [0, 1, 2]
    assert parse_float_list("0.01,0.05") == [0.01, 0.05]

    import pandas as pd

    per_seed = pd.DataFrame(
        [
            {"seed": 0, "alpha": 0.05, "method": "RAC", "miscov_a0": 0.1, "avg_utility": 0.2},
            {"seed": 1, "alpha": 0.05, "method": "RAC", "miscov_a0": 0.3, "avg_utility": 0.4},
        ]
    )
    agg = aggregate_seed_results(per_seed)
    row = agg.iloc[0]
    assert row["miscov_a0_mean"] == 0.2
    assert row["miscov_a0_count"] == 2


def test_acrac_toy_smoke():
    rng = np.random.default_rng(7)
    cal_probs = rng.dirichlet(np.ones(5), size=16)
    cal_labels = rng.integers(0, 5, size=16)
    test_probs = rng.dirichlet(np.ones(5), size=4)
    test_labels = rng.integers(0, 5, size=4)
    utility = MOVIELENS_UTILITY_MATRIX.copy()
    utility[:, 0] = 0.1 * (3 - np.arange(1, 6))

    result = run_acrac(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        0.05,
        [0, 1],
        make_utility_fn(utility),
        max_iter=3,
        eta0=1.0,
    )
    assert len(result["sets"]) == len(test_labels)
    assert set(result["action_frequencies"]) == {0, 1}


def test_vectorized_acrac_matches_literal_loop():
    rng = np.random.default_rng(17)
    cal_probs = rng.dirichlet(np.ones(5), size=12)
    cal_labels = rng.integers(0, 5, size=12)
    test_probs = rng.dirichlet(np.ones(5), size=3)
    test_labels = rng.integers(0, 5, size=3)
    utility = MOVIELENS_UTILITY_MATRIX.copy()
    utility[:, 0] = 0.1 * (3 - np.arange(1, 6))
    utility_fn = make_utility_fn(utility)

    fast_sets, fast_acts, fast_certs = predict_acrac(
        cal_probs,
        cal_labels,
        test_probs,
        0.05,
        [0, 1],
        utility_fn,
        max_iter=4,
        eta0=1.0,
    )
    cal_cands = precompute_candidates(cal_probs, [0, 1], utility_fn, include_alpha=0.05)
    test_cands_all = precompute_candidates(test_probs, [0, 1], utility_fn, include_alpha=0.05)
    slow_sets, slow_acts, slow_certs = [], [], []
    for i, test_cands in enumerate(test_cands_all):
        final_set = set()
        for y in range(5):
            lambdas = calibrate_lambdas_from_candidates(
                cal_cands,
                cal_labels,
                0.05,
                [0, 1],
                max_iter=4,
                eta0=1.0,
                decay_start=10**9,
                early_stop=False,
                extra_cands=test_cands,
                extra_label=y,
            )
            _, _, _, c_set = _select_candidate(test_cands, lambdas, 0.95)
            if y in c_set:
                final_set.add(y)
        if not final_set:
            final_set = {int(np.argmax(test_probs[i]))}
        action, cert = select_action_max_min(final_set, [0, 1], utility_fn)
        slow_sets.append(final_set)
        slow_acts.append(action)
        slow_certs.append(cert)

    assert fast_sets == slow_sets
    assert np.array_equal(fast_acts, np.array(slow_acts))
    assert np.allclose(fast_certs, np.array(slow_certs))
    assert len(test_labels) == len(fast_sets)


def test_movielens_seed_cache_reuse(tmp_path):
    cal_probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    cal_labels = np.array([0, 1])
    test_probs = np.array([[0.6, 0.4]])
    test_labels = np.array([0])
    seed_dir = tmp_path / "seed_0"
    save_prediction_cache(seed_dir, cal_probs, cal_labels, test_probs, test_labels)

    loaded = get_or_create_predictions_for_seed(
        "missing.zip",
        0,
        epochs=0,
        batch_size=1,
        device=torch.device("cpu"),
        seed_cache_root=tmp_path,
    )
    assert all(np.array_equal(actual, expected) for actual, expected in zip(loaded, [cal_probs, cal_labels, test_probs, test_labels]))
