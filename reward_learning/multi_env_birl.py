import numpy as np
from agent.q_learning_agent import ValueIteration
import copy

def logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


class MultiEnvBIRL:
    """
    Bayesian IRL across multiple MDPs (shared feature space).
    - envs: list of MDPs
    - demos: [(env_idx, [(s,a), (s,a), ...]), ...]
    - beta: scalar or list of betas per env
    """
    def __init__(self, envs, demos, beta, epsilon=1e-4):
        # Deepcopy envs so that reward updates don't leak between runs
        self.envs = [copy.deepcopy(e) for e in envs]
        self.epsilon = float(epsilon)
        self.beta = np.asarray(beta if np.ndim(beta) else [beta]*len(envs), dtype=float)

        # Convert demos -> per-env list
        self.demos = [[] for _ in range(len(envs))]
        for env_idx, pairs in demos:
            self.demos[env_idx].extend(pairs)

        self.num_mcmc_dims = len(self.envs[0].feature_weights)
        self.chain = None
        self.likelihoods = None
        self.accept_rate = None
        self.map_sol = None

    # ---------- Core helpers ----------

    def _q_for_env(self, env_id, w):
        """Set reward, run VI fully fresh, return converged Q-values."""
        env = self.envs[env_id]
        env.set_feature_weights(w)

        # fresh value iteration every call (like single BIRL)
        val_iter = ValueIteration(env)
        val_iter.run_value_iteration(epsilon=self.epsilon)
        Q = val_iter.get_q_values()
        return Q

    def calc_ll(self, w):
        """Compute log-likelihood over all envs."""
        w = np.asarray(w, dtype=float)
        log_like = 0.0
        for i, demo in enumerate(self.demos):
            if not demo:
                continue

            Q = self._q_for_env(i, w)
            b = float(self.beta[i])
            ts = set(getattr(self.envs[i], "terminal_states", []) or [])

            for s, a in demo:
                if s in ts:
                    continue
                Z = logsumexp(b * Q[s])
                log_like += b * Q[s, a] - Z

        return log_like

    # ---------- MCMC core ----------

    def generate_proposal(self, old_sol, stdev, normalize=True):
        prop = old_sol + stdev * np.random.randn(len(old_sol))
        if normalize:
            n = np.linalg.norm(prop)
            if n > 0:
                prop = prop / n
        return prop

    def initial_solution(self):
        v = np.random.randn(self.num_mcmc_dims)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def run_mcmc(self, samples, stepsize, normalize=True, adaptive=False, rng_seed=None):
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        num_samples = int(samples)
        stdev = float(stepsize)
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
                if cur_ll > map_ll:
                    map_ll, map_sol = cur_ll, cur

            self.chain[t, :] = cur
            self.likelihoods[t] = cur_ll

            # optional adaptive tuning
            if adaptive:
                acc_hist.append(1 if accept else 0)
                if (t + 1) % horizon == 0:
                    acc_rate = np.mean(acc_hist[-horizon:])
                    stdev = max(1e-5, stdev + lr / np.sqrt(t + 1) * (acc_rate - accept_target))

        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol

    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        b = int(len(self.chain) * burn_frac)
        return np.mean(self.chain[b::skip_rate], axis=0)
