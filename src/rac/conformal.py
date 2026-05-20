from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np


UtilityFn = Callable[[int, int], float]


def _matrix(utility_fn: UtilityFn) -> np.ndarray:
    mat = getattr(utility_fn, "matrix", None)
    if mat is None:
        raise ValueError("utility_fn must be produced by make_utility_fn so it exposes .matrix")
    return np.asarray(mat, dtype=float)


def hbtheta_and_arg(t: float, actions: Iterable[int], f_x: np.ndarray, utility_fn: UtilityFn) -> tuple[float, int]:
    mat = _matrix(utility_fn)
    probs = np.asarray(f_x, dtype=float)
    best_u, best_a = -math.inf, None
    for a in actions:
        uv = mat[: len(probs), int(a)]
        for u_candidate in np.sort(np.unique(uv))[::-1]:
            if probs[uv >= u_candidate].sum() >= t:
                if float(u_candidate) > best_u:
                    best_u, best_a = float(u_candidate), int(a)
                break
    if best_a is None:
        best_a = int(next(iter(actions)))
    return best_u, best_a


def get_conformal_set(x_probs: np.ndarray, s_value: float, actions: list[int], utility_fn: UtilityFn) -> set[int]:
    mat = _matrix(utility_fn)
    _, best_a = hbtheta_and_arg(s_value, actions, x_probs, utility_fn)
    uv = mat[: len(x_probs), best_a]
    return {y for y in range(len(x_probs)) if np.asarray(x_probs)[uv > uv[y]].sum() <= s_value}


def select_action_max_min(s_indices: set[int], actions: list[int], utility_fn: UtilityFn) -> tuple[int, float]:
    mat = _matrix(utility_fn)
    best_action, best_min_util = actions[0], float("-inf")
    for a in actions:
        min_util = min(float(mat[y, a]) for y in s_indices)
        if min_util > best_min_util:
            best_min_util, best_action = min_util, int(a)
    return best_action, best_min_util


def precompute_candidates(
    probs_array: np.ndarray,
    actions: list[int],
    utility_fn: UtilityFn,
    include_alpha: float | None = None,
) -> list[list[tuple[float, float, int, frozenset[int]]]]:
    mat = _matrix(utility_fn)
    num_labels = mat.shape[0]
    all_candidates = []
    for xp in probs_array:
        s_set = {0.0, 1.0}
        if include_alpha is not None:
            s_set.add(1.0 - include_alpha)

        action_pairs: dict[int, list[tuple[float, float]]] = {}
        for a in actions:
            uv = mat[:num_labels, a]
            pairs = []
            for u_cand in np.sort(np.unique(uv))[::-1]:
                t_val = float(xp[uv >= u_cand].sum())
                pairs.append((float(u_cand), t_val))
                s_set.add(t_val)
            action_pairs[a] = pairs

        candidates = []
        for s in sorted(s_set):
            best_u, best_a_idx = -math.inf, 0
            for local_idx, a in enumerate(actions):
                for u_cand, t_val in action_pairs[a]:
                    if t_val >= s:
                        if u_cand > best_u:
                            best_u, best_a_idx = u_cand, local_idx
                        break
            uv = mat[:num_labels, actions[best_a_idx]]
            c_set = frozenset(y for y in range(num_labels) if xp[uv > uv[y]].sum() <= s)
            candidates.append((float(s), float(best_u), int(best_a_idx), c_set))
        all_candidates.append(candidates)
    return all_candidates


def find_threshold_q(calib_probs: np.ndarray, calib_labels: np.ndarray, alpha: float, actions: list[int], utility_fn: UtilityFn) -> float:
    all_cands = precompute_candidates(calib_probs, actions, utility_fn)
    n = len(all_cands)

    def coverage(beta: float) -> float:
        hits = 0
        for i in range(n):
            best_score, best_idx = -math.inf, 0
            for j, (s, theta_u, _, _) in enumerate(all_cands[i]):
                score = beta * s + theta_u
                if score > best_score:
                    best_score, best_idx = score, j
            if int(calib_labels[i]) in all_cands[i][best_idx][3]:
                hits += 1
        return hits / n

    beta_now, beta_past = 1.0, 0.0
    while coverage(beta_now) < 1.0 - alpha:
        beta_past = beta_now
        beta_now *= 2.0
    for _ in range(20):
        beta_mean = (beta_past + beta_now) / 2.0
        if coverage(beta_mean) >= 1.0 - alpha:
            beta_now = beta_mean
        else:
            beta_past = beta_mean
    return (beta_past + beta_now) / 2.0


def predict_rac(test_probs: np.ndarray, q: float, actions: list[int], utility_fn: UtilityFn) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    all_cands = precompute_candidates(test_probs, actions, utility_fn)
    sets_list, acts_list, certs_list = [], [], []
    for i, cands in enumerate(all_cands):
        best_score, best_idx = -math.inf, 0
        for j, (s, theta_u, _, _) in enumerate(cands):
            score = q * s + theta_u
            if score > best_score:
                best_score, best_idx = score, j
        _, cert, a_idx, c_set = cands[best_idx]
        sets_list.append(set(c_set) if c_set else {int(np.argmax(test_probs[i]))})
        acts_list.append(actions[a_idx])
        certs_list.append(cert)
    return sets_list, np.array(acts_list), np.array(certs_list)


def compute_conformal_threshold_tail(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    scores = []
    for probs, true_label in zip(cal_probs, cal_labels):
        p_true = probs[int(true_label)]
        mask = probs > p_true
        mask[int(true_label)] = False
        scores.append(float(probs[mask].sum()))
    return float(np.quantile(scores, 1.0 - alpha, method="higher"))


def predict_tail_score(test_probs: np.ndarray, threshold: float, actions: list[int], utility_fn: UtilityFn) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    sets_list, acts_list, certs_list = [], [], []
    for probs in test_probs:
        scores = np.zeros(len(probs))
        for c in range(len(probs)):
            mask = probs > probs[c]
            mask[c] = False
            scores[c] = probs[mask].sum()
        s_idx = set(np.where(scores <= threshold)[0])
        if not s_idx:
            s_idx = {int(np.argmax(probs))}
        a_star, mm_val = select_action_max_min(s_idx, actions, utility_fn)
        sets_list.append(s_idx)
        acts_list.append(a_star)
        certs_list.append(mm_val)
    return sets_list, np.array(acts_list), np.array(certs_list)


def compute_conformal_threshold_one_minus_true(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    scores = [1.0 - float(cal_probs[i, int(cal_labels[i])]) for i in range(len(cal_probs))]
    return float(np.quantile(scores, 1.0 - alpha, method="higher"))


def predict_one_minus_true_score(test_probs: np.ndarray, threshold: float, actions: list[int], utility_fn: UtilityFn) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    cutoff = 1.0 - threshold
    sets_list, acts_list, certs_list = [], [], []
    for probs in test_probs:
        s_idx = set(np.where(probs >= cutoff)[0])
        if not s_idx:
            s_idx = {int(np.argmax(probs))}
        a_star, mm_val = select_action_max_min(s_idx, actions, utility_fn)
        sets_list.append(s_idx)
        acts_list.append(a_star)
        certs_list.append(mm_val)
    return sets_list, np.array(acts_list), np.array(certs_list)
