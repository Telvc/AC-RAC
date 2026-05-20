from __future__ import annotations

import math

import numpy as np

from .conformal import UtilityFn, precompute_candidates, select_action_max_min


Candidate = tuple[float, float, int, frozenset[int]]
CandidateArrays = dict[str, np.ndarray]


def _select_candidate(cands: list[Candidate], lambdas: np.ndarray, one_minus_alpha: float) -> Candidate:
    best_score, best_idx = -math.inf, 0
    for j, (s, theta_u, a_idx, _) in enumerate(cands):
        score = theta_u + lambdas[a_idx] * (s - one_minus_alpha)
        if score > best_score:
            best_score, best_idx = score, j
    return cands[best_idx]


def _candidate_arrays(all_cands: list[list[Candidate]], num_labels: int) -> CandidateArrays:
    n = len(all_cands)
    max_candidates = max((len(cands) for cands in all_cands), default=0)
    s_values = np.zeros((n, max_candidates), dtype=float)
    theta_values = np.zeros((n, max_candidates), dtype=float)
    action_indices = np.zeros((n, max_candidates), dtype=int)
    contains = np.zeros((n, max_candidates, num_labels), dtype=bool)
    valid = np.zeros((n, max_candidates), dtype=bool)

    for i, cands in enumerate(all_cands):
        for j, (s, theta_u, a_idx, c_set) in enumerate(cands):
            s_values[i, j] = float(s)
            theta_values[i, j] = float(theta_u)
            action_indices[i, j] = int(a_idx)
            valid[i, j] = True
            for label in c_set:
                contains[i, j, int(label)] = True

    return {
        "s": s_values,
        "theta": theta_values,
        "action": action_indices,
        "contains": contains,
        "valid": valid,
    }


def _slice_candidate_arrays(arrays: CandidateArrays, idx: int) -> CandidateArrays:
    return {key: value[idx : idx + 1] for key, value in arrays.items()}


def _select_candidate_arrays(
    arrays: CandidateArrays,
    lambdas: np.ndarray,
    one_minus_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lambdas = np.asarray(lambdas, dtype=float)
    squeeze = False
    if lambdas.ndim == 1:
        lambdas = lambdas[None, :]
        squeeze = True

    delta = arrays["s"] - one_minus_alpha
    lambda_terms = lambdas[:, arrays["action"]]
    scores = arrays["theta"][None, :, :] + lambda_terms * delta[None, :, :]
    scores = np.where(arrays["valid"][None, :, :], scores, -np.inf)
    best = np.argmax(scores, axis=2)
    row_idx = np.arange(arrays["s"].shape[0])[None, :]
    selected_actions = arrays["action"][row_idx, best]
    selected_theta = arrays["theta"][row_idx, best]
    selected_contains = arrays["contains"][row_idx, best, :]

    if squeeze:
        return selected_actions[0], selected_theta[0], selected_contains[0], best[0]
    return selected_actions, selected_theta, selected_contains, best


def _coverage_counts_arrays(
    arrays: CandidateArrays,
    labels: np.ndarray,
    lambdas: np.ndarray,
    one_minus_alpha: float,
    num_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_actions, _, selected_contains, _ = _select_candidate_arrays(arrays, lambdas, one_minus_alpha)
    labels = np.asarray(labels, dtype=int)
    if np.asarray(lambdas).ndim == 1:
        hits_by_row = selected_contains[np.arange(len(labels)), labels]
        counts = np.bincount(selected_actions.astype(int), minlength=num_actions).astype(float)
        hits = np.bincount(selected_actions.astype(int), weights=hits_by_row.astype(float), minlength=num_actions)
        return counts, hits

    hits_by_row = selected_contains[:, np.arange(len(labels)), labels]
    counts = np.zeros((np.asarray(lambdas).shape[0], num_actions), dtype=float)
    hits = np.zeros_like(counts)
    for a_idx in range(num_actions):
        mask = selected_actions == a_idx
        counts[:, a_idx] = mask.sum(axis=1)
        hits[:, a_idx] = (hits_by_row & mask).sum(axis=1)
    return counts, hits


def _coverage_counts(
    all_cands: list[list[Candidate]],
    labels: np.ndarray,
    lambdas: np.ndarray,
    one_minus_alpha: float,
    num_actions: int,
    extra_cands: list[Candidate] | None = None,
    extra_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(num_actions, dtype=float)
    hits = np.zeros(num_actions, dtype=float)
    selected_actions = np.zeros(len(all_cands) + int(extra_cands is not None), dtype=int)

    for i, cands in enumerate(all_cands):
        _, _, a_idx, c_set = _select_candidate(cands, lambdas, one_minus_alpha)
        y_true = int(labels[i])
        counts[a_idx] += 1.0
        hits[a_idx] += float(y_true in c_set)
        selected_actions[i] = a_idx

    if extra_cands is not None:
        if extra_label is None:
            raise ValueError("extra_label is required when extra_cands is provided")
        _, _, a_idx, c_set = _select_candidate(extra_cands, lambdas, one_minus_alpha)
        counts[a_idx] += 1.0
        hits[a_idx] += float(int(extra_label) in c_set)
        selected_actions[-1] = a_idx

    return counts, hits, selected_actions


def calibrate_lambdas_from_candidates(
    all_cands: list[list[Candidate]],
    labels: np.ndarray,
    alpha: float,
    actions: list[int],
    max_iter: int = 400,
    eta0: float = 25.0,
    decay_start: int = 350,
    min_iter_before_stop: int = 80,
    tol: float = 1e-3,
    early_stop: bool = True,
    initial_lambdas: np.ndarray | None = None,
    extra_cands: list[Candidate] | None = None,
    extra_label: int | None = None,
) -> np.ndarray:
    lambdas = np.zeros(len(actions), dtype=float)
    if initial_lambdas is not None:
        lambdas = np.asarray(initial_lambdas, dtype=float).copy()
    one_minus_alpha = 1.0 - alpha
    n = len(all_cands) + int(extra_cands is not None)

    for k in range(1, max_iter + 1):
        counts, hits, _ = _coverage_counts(
            all_cands,
            labels,
            lambdas,
            one_minus_alpha,
            len(actions),
            extra_cands=extra_cands,
            extra_label=extra_label,
        )

        coverage_arr = np.divide(hits, counts, out=np.ones_like(hits), where=counts > 0)
        if early_stop and k >= min_iter_before_stop and coverage_arr.min() >= one_minus_alpha - tol:
            break
        step = eta0 if k <= decay_start else eta0 / math.sqrt(k - decay_start)
        grads = (counts / max(n, 1)) * (coverage_arr - one_minus_alpha)
        lambdas = np.maximum(0.0, lambdas - step * grads)
    return lambdas


def predict_acrac(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
    actions: list[int],
    utility_fn: UtilityFn,
    max_iter: int = 200,
    eta0: float = 30.0,
    decay_start: int = 10**9,
    warm_start_lambdas: np.ndarray | None = None,
    progress_label: str | None = None,
    progress_every: int | None = None,
    batch_size: int = 1,
) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    """Predict with label-wise AC-RAC.

    For each test point and each candidate label y, this solves Algorithm 2
    on the calibration set augmented with (x_test, y). The final prediction
    set contains exactly the labels accepted by their label-wise lambdas.
    """
    num_labels = np.asarray(utility_fn.matrix).shape[0]  # type: ignore[attr-defined]
    calib_cands = precompute_candidates(calib_probs, actions, utility_fn, include_alpha=alpha)
    test_cands_all = precompute_candidates(test_probs, actions, utility_fn, include_alpha=alpha)
    calib_arrays = _candidate_arrays(calib_cands, num_labels)
    test_arrays = _candidate_arrays(test_cands_all, num_labels)
    one_minus_alpha = 1.0 - alpha
    sets_list, acts_list, certs_list = [], [], []

    if progress_label:
        print(f"[AC-RAC] {progress_label}: start n_cal={len(calib_labels)} n_test={len(test_probs)} K={max_iter}", flush=True)
    batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(test_cands_all), batch_size):
        batch_end = min(batch_start + batch_size, len(test_cands_all))
        crossed_progress = progress_every and (batch_end // progress_every > batch_start // progress_every)
        if progress_label and progress_every and (batch_start == 0 or crossed_progress or batch_end == len(test_probs)):
            print(f"[AC-RAC] {progress_label}: test {batch_end}/{len(test_probs)}", flush=True)
        batch_arrays = {key: value[batch_start:batch_end] for key, value in test_arrays.items()}
        current_batch_size = batch_end - batch_start
        if warm_start_lambdas is None:
            lambdas = np.zeros((current_batch_size, num_labels, len(actions)), dtype=float)
        else:
            lambdas = np.tile(np.asarray(warm_start_lambdas, dtype=float), (current_batch_size, num_labels, 1))
        row_for_candidate = np.repeat(np.arange(current_batch_size), num_labels)
        label_for_candidate = np.tile(np.arange(num_labels), current_batch_size)
        flat_index = np.arange(current_batch_size * num_labels)

        for k in range(1, max_iter + 1):
            flat_lambdas = lambdas.reshape(current_batch_size * num_labels, len(actions))
            counts, hits = _coverage_counts_arrays(calib_arrays, calib_labels, flat_lambdas, one_minus_alpha, len(actions))
            test_actions, _, test_contains, _ = _select_candidate_arrays(batch_arrays, flat_lambdas, one_minus_alpha)
            paired_actions = test_actions[flat_index, row_for_candidate].astype(int)
            paired_hits = test_contains[flat_index, row_for_candidate, label_for_candidate]
            counts[flat_index, paired_actions] += 1.0
            hits[flat_index, paired_actions] += paired_hits.astype(float)

            coverage_arr = np.divide(hits, counts, out=np.ones_like(hits), where=counts > 0)
            step = eta0 if k <= decay_start else eta0 / math.sqrt(k - decay_start)
            grads = (counts / (len(calib_labels) + 1.0)) * (coverage_arr - one_minus_alpha)
            lambdas = np.maximum(0.0, flat_lambdas - step * grads).reshape(current_batch_size, num_labels, len(actions))

        flat_lambdas = lambdas.reshape(current_batch_size * num_labels, len(actions))
        _, _, test_contains, _ = _select_candidate_arrays(batch_arrays, flat_lambdas, one_minus_alpha)
        accepted = test_contains[flat_index, row_for_candidate, label_for_candidate].reshape(current_batch_size, num_labels)

        for local_idx, accepted_row in enumerate(accepted):
            test_idx = batch_start + local_idx
            final_set = {int(label) for label in np.where(accepted_row)[0]}
            if not final_set:
                final_set = {int(np.argmax(test_probs[test_idx]))}
            action, cert = select_action_max_min(final_set, actions, utility_fn)
            sets_list.append(final_set)
            acts_list.append(action)
            certs_list.append(cert)

    if progress_label:
        print(f"[AC-RAC] {progress_label}: done", flush=True)
    return sets_list, np.array(acts_list, dtype=int), np.array(certs_list, dtype=float)
