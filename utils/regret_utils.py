import numpy as np
from .common_helper import calculate_expected_value_difference
from .successor_features import build_Pi_from_q

# def regrets_from_Q(envs, Q_list, *, tie_eps=1e-10, epsilon=1e-4, normalize_with_random_policy=False):
#     regrets = []
#     for env, Q in zip(envs, Q_list):
#         pi = build_Pi_from_q(env, Q, tie_eps=tie_eps)
#         r = calculate_expected_value_difference(
#             eval_policy=pi,
#             env=env,
#             epsilon=epsilon,
#             normalize_with_random_policy=normalize_with_random_policy,
#         )
#         regrets.append(float(r))
#     return np.asarray(regrets, float)


# def compare_regret_from_Q(envs, Q_scot_list, Q_rand_list, *,
#                           tie_eps=1e-10, epsilon=1e-4, normalize_with_random_policy=False):

#     reg_scot = regrets_from_Q(envs, Q_scot_list,
#                               tie_eps=tie_eps,
#                               epsilon=epsilon,
#                               normalize_with_random_policy=normalize_with_random_policy)
#     reg_rand = regrets_from_Q(envs, Q_rand_list,
#                               tie_eps=tie_eps,
#                               epsilon=epsilon,
#                               normalize_with_random_policy=normalize_with_random_policy)

#     def _stats(x):
#         return {
#             "mean": float(np.mean(x)),
#             "std": float(np.std(x)),
#             "median": float(np.median(x)),
#             "min": float(np.min(x)),
#             "max": float(np.max(x)),
#         }

#     return {
#         "per_env": {
#             "SCOT": reg_scot,
#             "RandomOpt": reg_rand,
#             "Delta": reg_scot - reg_rand,
#         },
#         "summary": {
#             "SCOT": _stats(reg_scot),
#             "RandomOpt": _stats(reg_rand),
#             "delta_mean": float(np.mean(reg_scot) - np.mean(reg_rand)),
#         },
#         "stacked_table": np.stack([reg_scot, reg_rand], axis=1),
#     }


import numpy as np
from agent.q_learning_agent_ import ValueIteration
from utils.successor_features import build_Pi_from_q
from utils.common_helper import calculate_expected_value_difference


def compute_Q_from_weights_with_VI(env, w, vi_epsilon=1e-6):
    """Set env weights to w, run VI, return Q; then restore old weights."""
    old_w = np.array(env.feature_weights, copy=True)

    env.set_feature_weights(w)
    VI = ValueIteration(env)
    VI.run_value_iteration(epsilon=vi_epsilon)
    Q = VI.get_q_values()

    env.set_feature_weights(old_w)
    return Q

def regrets_from_Q(envs, Q_list, *, tie_eps=1e-10, epsilon=1e-4, normalize_with_random_policy=False):
    regrets = []
    for env, Q in zip(envs, Q_list):
        pi = build_Pi_from_q(env, Q, tie_eps=tie_eps)
        r = calculate_expected_value_difference(
            eval_policy=pi,
            env=env,
            epsilon=epsilon,
            normalize_with_random_policy=normalize_with_random_policy,
        )
        regrets.append(float(r))
    return np.asarray(regrets, float)
