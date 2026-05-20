from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from .acrac import predict_acrac
from .conformal import (
    UtilityFn,
    compute_conformal_threshold_one_minus_true,
    compute_conformal_threshold_tail,
    find_threshold_q,
    predict_one_minus_true_score,
    predict_rac,
    predict_tail_score,
)


def summarize_predictions(
    sets_list: list[set[int]],
    acts: np.ndarray,
    certs: np.ndarray,
    true_labels: np.ndarray,
    actions: list[int],
    utility_fn: UtilityFn,
    critical_error_fn: Callable[[int, int, float], bool] | None = None,
) -> dict:
    n = len(true_labels)
    per_action_hits = {a: 0 for a in actions}
    per_action_count = {a: 0 for a in actions}
    per_action_sizes = {a: [] for a in actions}
    per_action_fdr = {a: [] for a in actions}
    realized_utils, set_sizes, fdr_values = [], [], []
    cover_hits = 0
    critical_errors = 0

    for i in range(n):
        c_set = sets_list[i]
        action = int(acts[i])
        y_true = int(true_labels[i])
        utility = float(utility_fn(action, y_true))
        realized_utils.append(utility)
        if critical_error_fn is not None:
            critical_errors += int(critical_error_fn(action, y_true, utility))
        else:
            critical_errors += int(utility == 0)

        covered = y_true in c_set
        cover_hits += int(covered)
        per_action_count[action] += 1
        per_action_hits[action] += int(covered)

        size = len(c_set)
        set_sizes.append(size)
        per_action_sizes[action].append(size)
        fdr = (size - int(covered)) / max(size, 1)
        fdr_values.append(fdr)
        per_action_fdr[action].append(fdr)

    coverage_by_action = {
        a: per_action_hits[a] / per_action_count[a] if per_action_count[a] > 0 else np.nan
        for a in actions
    }
    set_size_by_action = {
        a: float(np.mean(per_action_sizes[a])) if per_action_sizes[a] else np.nan
        for a in actions
    }
    fdr_by_action = {
        a: float(np.mean(per_action_fdr[a])) if per_action_fdr[a] else np.nan
        for a in actions
    }
    action_counts = dict(Counter(int(a) for a in acts))

    return {
        "coverage_overall": cover_hits / n if n else np.nan,
        "coverage_by_action": coverage_by_action,
        "action_counts": action_counts,
        "action_frequencies": {a: action_counts.get(a, 0) / n if n else 0.0 for a in actions},
        "sets": [set(s) for s in sets_list],
        "set_size_mean": float(np.mean(set_sizes)) if set_sizes else np.nan,
        "set_size_by_action": set_size_by_action,
        "avg_utility": float(np.mean(realized_utils)) if realized_utils else np.nan,
        "avg_maxmin": float(np.mean(certs)) if len(certs) else np.nan,
        "fdr_overall": float(np.mean(fdr_values)) if fdr_values else np.nan,
        "fdr_by_action": fdr_by_action,
        "critical_error_rate": critical_errors / n if n else np.nan,
        "acts": np.array(acts, dtype=int),
    }


def run_acrac(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn, **kwargs) -> dict:
    sets_list, acts, certs = predict_acrac(cal_p, cal_l, test_p, alpha, actions, utility_fn, **kwargs)
    return summarize_predictions(sets_list, acts, certs, test_l, actions, utility_fn)


def run_rac(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn) -> dict:
    q = find_threshold_q(cal_p, cal_l, alpha, actions, utility_fn)
    sets_list, acts, certs = predict_rac(test_p, q, actions, utility_fn)
    result = summarize_predictions(sets_list, acts, certs, test_l, actions, utility_fn)
    result["q"] = q
    return result


def run_score_tail(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn) -> dict:
    threshold = compute_conformal_threshold_tail(cal_p, cal_l, alpha)
    sets_list, acts, certs = predict_tail_score(test_p, threshold, actions, utility_fn)
    result = summarize_predictions(sets_list, acts, certs, test_l, actions, utility_fn)
    result["threshold"] = threshold
    return result


def run_score_one_minus_true(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn) -> dict:
    threshold = compute_conformal_threshold_one_minus_true(cal_p, cal_l, alpha)
    sets_list, acts, certs = predict_one_minus_true_score(test_p, threshold, actions, utility_fn)
    result = summarize_predictions(sets_list, acts, certs, test_l, actions, utility_fn)
    result["threshold"] = threshold
    return result


def run_best_response(test_p: np.ndarray, test_l: np.ndarray, actions: list[int], utility_fn: UtilityFn) -> dict:
    mat = getattr(utility_fn, "matrix")
    acts, realized_utils = [], []
    for probs, y_true in zip(test_p, test_l):
        expected = np.asarray(probs) @ np.asarray(mat)
        best_local = int(np.argmax([expected[a] for a in actions]))
        action = actions[best_local]
        acts.append(action)
        realized_utils.append(float(utility_fn(action, int(y_true))))
    counts = dict(Counter(acts))
    n = len(test_l)
    return {
        "avg_utility": float(np.mean(realized_utils)),
        "action_counts": counts,
        "action_frequencies": {a: counts.get(a, 0) / n for a in actions},
        "acts": np.array(acts, dtype=int),
    }


METHOD_ORDER = ["AC-RAC", "RAC", "Score-1", "Score-2"]
METHOD_COLORS = {
    "AC-RAC": "#d62728",
    "RAC": "#1f77b4",
    "Score-1": "#ff7f0e",
    "Score-2": "#2ca02c",
    "Best Response": "#000000",
}
METHOD_MARKERS = {"AC-RAC": "o", "RAC": "s", "Score-1": "^", "Score-2": "d"}


def run_all_methods(
    cal_p,
    cal_l,
    test_p,
    test_l,
    alpha,
    actions,
    utility_fn,
    ac_kwargs: dict | None = None,
) -> dict[str, dict]:
    ac_kwargs = ac_kwargs or {}
    ac_keys = {
        "max_iter",
        "eta0",
        "decay_start",
        "warm_start_lambdas",
        "progress_label",
        "progress_every",
        "batch_size",
    }
    acrac_kwargs = {key: value for key, value in ac_kwargs.items() if key in ac_keys}
    ac_result = run_acrac(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn, **acrac_kwargs)
    return {
        "AC-RAC": ac_result,
        "RAC": run_rac(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn),
        "Score-1": run_score_tail(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn),
        "Score-2": run_score_one_minus_true(cal_p, cal_l, test_p, test_l, alpha, actions, utility_fn),
    }
