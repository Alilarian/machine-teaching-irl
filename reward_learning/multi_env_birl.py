import numpy as np
import copy
from agent.q_learning_agent import ValueIteration

def logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

class MultiEnvBIRL:
    """
    Bayesian IRL over a *family* of MDPs that share the same feature space.
    - envs:   list of MDPs
    - demos:  list of lists; demos[i] is a list of (s,a) from envs[i]
              (Alternatively, pass a flat list of (i, s, a) triples; see the helper below.)
    - beta:   scalar or array-like of length len(envs)
    """
    def __init__(self, envs, demos, beta, epsilon=1e-4):
        self.envs = [copy.deepcopy(e) for e in envs]
        self.demos = demos
        self.epsilon = epsilon
        self.beta = np.asarray(beta if np.ndim(beta) else [beta]*len(envs), dtype=float)
        # assume all envs share the same feature vector length
        self.num_mcmc_dims = len(self.envs[0].feature_weights)
        # per-env VI handles
        self._VIs = [ValueIteration(e) for e in self.envs]
        # cache: (env_id, reward_key) -> Q
        self._q_cache = {}

    @staticmethod
    def demos_from_triples(triples, num_envs):
        """Convert flat [(i,s,a), ...] into per-env list-of-(s,a)."""
        out = [[] for _ in range(num_envs)]
        for i, s, a in triples:
            out[int(i)].append((int(s), int(a)))
        return out

    def _reward_key(self, w):
        # hashing key for cache; normalize to unit norm to avoid scale confounds if you normalize proposals
        w = np.asarray(w, dtype=float)
        if not np.isfinite(np.linalg.norm(w)) or np.linalg.norm(w) == 0:
            return tuple(np.zeros_like(w))
        wn = w / np.linalg.norm(w)
        return tuple(np.round(wn, 12))

    def _q_for_env(self, env_id, w):
        key = (env_id, self._reward_key(w))
        if key in self._q_cache:
            return self._q_cache[key]
        env = self.envs[env_id]
        env.set_feature_weights(w)
        q_vals = self._VIs[env_id].get_q_values()
        self._q_cache[key] = q_vals
        return q_vals

    def calc_ll(self, w):
        """Sum log-likelihood across all envs and their demonstrations."""
        log_like = 0.0  # uniform prior
        for i, demo_i in enumerate(self.demos):
            if not demo_i:
                continue
            Q = self._q_for_env(i, w)
            b = self.beta[i]
            for s, a in demo_i:
                if hasattr(self.envs[i], "terminal_states") and s in set(self.envs[i].terminal_states or []):
                    continue
                Z = logsumexp(b * Q[s])          # log-partition at state s
                log_like += b * Q[s, a] - Z
        return log_like

    # --- MH machinery (unchanged API) ---
    def generate_proposal(self, old_sol, stdev, normalize=True):
        prop = old_sol + stdev * np.random.randn(len(old_sol))
        if normalize:
            n = np.linalg.norm(prop)
            if n > 0: prop = prop / n
        return prop

    def initial_solution(self):
        v = np.random.randn(self.num_mcmc_dims)
        v /= np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        return v

    def run_mcmc(self, samples, stepsize, normalize=True, adaptive=False):
        num_samples = samples
        stdev = stepsize
        accept_cnt = 0

        accept_target = 0.4
        horizon = max(1, num_samples // 100)
        lr = 0.05
        acc_hist = []

        self.chain = np.zeros((num_samples, self.num_mcmc_dims))
        self.likelihoods = np.zeros(num_samples)

        cur = self.initial_solution()
        cur_ll = self.calc_ll(cur)
        map_ll, map_sol = cur_ll, cur

        for t in range(num_samples):
            prop = self.generate_proposal(cur, stdev, normalize=normalize)
            prop_ll = self.calc_ll(prop)

            accept = (prop_ll > cur_ll) or (np.random.rand() < np.exp(prop_ll - cur_ll))
            if accept:
                cur, cur_ll = prop, prop_ll
                accept_cnt += 1
                acc_hist.append(1)
                if cur_ll > map_ll:
                    map_ll, map_sol = cur_ll, cur
            else:
                acc_hist.append(0)

            self.chain[t, :] = cur
            self.likelihoods[t] = cur_ll

            if adaptive and (t+1) % horizon == 0:
                acc_rate = np.mean(acc_hist[-horizon:])
                stdev = max(1e-5, stdev + lr/np.sqrt(t+1) * (acc_rate - accept_target))

        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol

    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        b = int(len(self.chain) * burn_frac)
        return np.mean(self.chain[b::skip_rate], axis=0)
