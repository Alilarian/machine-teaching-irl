import numpy as np
import copy
from agent.q_learning_agent_ import ValueIteration
from utils.common_helper import compute_reward_for_trajectory
from scipy.special import logsumexp
from numba import njit
from tqdm import tqdm

@njit
def demo_ll_numba(Q, traj, beta, terminal_mask):
    """
    Fast likelihood for demonstration feedback:
        sum_t [ beta*Q[s_t, a_t] - logsumexp(beta*Q[s_t]) ]
    """
    log_l = 0.0
    n_states, n_actions = Q.shape

    for t in range(len(traj)):
        s = traj[t][0]
        a = traj[t][1]

        if terminal_mask[s] or a < 0:
            continue

        row = beta * Q[s]

        # logsumexp
        m = np.max(row)
        Z = m + np.log(np.sum(np.exp(row - m)))

        log_l += beta * Q[s, a] - Z

    return log_l


class MultiEnvAtomicBIRL:
    """
    Unified Bayesian IRL supporting:
        - multiple MDPs
        - SCOT atoms in canonical format:
              atoms_flat = [(env_idx, Atom), (env_idx, Atom), ...]
        - feedback types:
              'demo', 'pairwise', 'estop', 'improvement'
    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(
        self,
        envs,
        atoms_flat,          # << THE ONLY SUPPORTED INPUT FORMAT
        *,
        beta_demo=5.0,
        beta_pairwise=1.0,
        beta_estop=1.0,
        beta_improvement=1.0,
        epsilon=1e-4
    ):
        """
        envs: list of environments
        atoms_flat: list of (env_idx, Atom) pairs
            env_idx must be 0 <= env_idx < len(envs)
        """

        # Deep copy envs so reward reuse does not leak between runs
        self.envs = [copy.deepcopy(e) for e in envs]
        self.epsilon = float(epsilon)
        num_envs = len(envs)

        # ------------------------------------------------------------
        # Convert atoms_flat -> atoms_per_env (list of lists of Atom)
        # ------------------------------------------------------------
        atoms_per_env = [[] for _ in range(num_envs)]

        for env_idx, atom in atoms_flat:
            if env_idx < 0 or env_idx >= num_envs:
                raise ValueError(f"Invalid env_idx={env_idx} in atoms_flat.")
            atoms_per_env[env_idx].append(atom)

        self.atoms_per_env = atoms_per_env

        # ============================================================
        # Precompute Successor Features for each environment
        # ============================================================
        from utils.successor_features import compute_successor_features_iterative_from_q

        self.mu_sa_per_env = []

        for env in self.envs:
            # Compute SFs using a random reward vector or zero vector
            # (policy doesn't matter — SFs encode discounted dynamics)
            #dummy_w = np.zeros_like(env.feature_weights)
            dummy_w = np.random.randn(len(env.feature_weights)) * 0.01
            env.set_feature_weights(dummy_w)

            # Perform a single VI to obtain policy π under dummy_w
            vi = ValueIteration(env)
            vi.run_value_iteration(epsilon=self.epsilon)
            Q = vi.get_q_values()

            # Compute SFs under the induced deterministic π
            mu_sa, mu_s, Phi, P_pi = compute_successor_features_iterative_from_q(
                env, Q,
            )
            self.mu_sa_per_env.append(mu_sa)


        # Feature dimension from first env
        self.num_mcmc_dims = len(self.envs[0].feature_weights)

        # Store β parameters
        self.beta_demo = beta_demo
        self.beta_pairwise = beta_pairwise
        self.beta_estop = beta_estop
        self.beta_improvement = beta_improvement

        # MCMC state
        self.chain = None
        self.likelihoods = None
        self.map_sol = None
        self.accept_rate = None

        # Determine which envs need Q(s,a)
        self.needs_q = [False] * num_envs
        for i, atoms in enumerate(self.atoms_per_env):
            for atom in atoms:
                if atom.feedback_type == "demo":
                    self.needs_q[i] = True
                    break

    # ------------------------------------------------------------
    # Likelihood evaluation for a reward vector
    # ------------------------------------------------------------
    def calc_ll(self, w):
        """
        Compute log-likelihood of all atoms across all environments.

        w: reward weight vector
        """
        w = np.asarray(w, float)
        total = 0.0

        for env_idx, env in enumerate(self.envs):
            atoms = self.atoms_per_env[env_idx]
            if not atoms:
                continue

            # Set reward
            env.set_feature_weights(w)

            # Compute Q only if needed
            Q = None
            # if self.needs_q[env_idx]:
            #     vi = ValueIteration(env)
            #     vi.run_value_iteration(epsilon=self.epsilon)
            #     Q = vi.get_q_values()
            if self.needs_q[env_idx]:
                # SUCCESSOR FEATURE Q-VALUE COMPUTATION
                mu_sa = self.mu_sa_per_env[env_idx]      # (S,A,d)
                Q = np.tensordot(mu_sa, w, axes=(2,0))   # → shape (S,A)


            # Evaluate each atom
            for atom in atoms:
                ft = atom.feedback_type
                data = atom.data

                if ft == "demo":
                    total += self._ll_demo(env, Q, data)

                elif ft == "pairwise":
                    total += self._ll_pairwise(env, data)

                elif ft == "estop":
                    total += self._ll_estop(env, data)

                elif ft == "improvement":
                    total += self._ll_improvement(env, data)

                else:
                    raise ValueError(f"Unknown feedback type: {ft}")

        return float(total)

    # ------------------------------------------------------------
    # Likelihood models
    # ------------------------------------------------------------

    # (1) DEMO: data = list[(s,a)]
    # def _ll_demo(self, env, Q, traj):
    #     if Q is None:
    #         raise RuntimeError("Demo likelihood requires Q-values.")

    #     beta = self.beta_demo
    #     ts = set(env.terminal_states or [])
    #     log_l = 0.0

    #     for s, a in traj:
    #         if s in ts or a is None:
    #             continue
    #         Z = logsumexp(beta * Q[s])
    #         log_l += beta * Q[s, a] - Z

    #     return log_l
    def _ll_demo(self, env, Q, traj):
        if Q is None:
            raise RuntimeError("Demo likelihood requires Q-values.")

        beta = self.beta_demo

        # Convert traj list[(s, a)] → ndarray
        traj_arr = np.array(traj, dtype=np.int32)

        # Build a quick terminal mask (NumPy)
        ts = env.terminal_states or []
        terminal_mask = np.zeros(Q.shape[0], dtype=np.bool_)
        for s in ts:
            terminal_mask[s] = True

        # Numba kernel
        return float(demo_ll_numba(Q, traj_arr, beta, terminal_mask))



    # (2) PAIRWISE: data = (traj1, traj2)
    def _ll_pairwise(self, env, pair):
        beta = self.beta_pairwise
        traj1, traj2 = pair

        r1 = compute_reward_for_trajectory(env, traj1)
        r2 = compute_reward_for_trajectory(env, traj2)

        Z = logsumexp([beta * r1, beta * r2])
        return beta * r1 - Z

    # (3) E-STOP: data = (trajectory, t_stop)
    def _ll_estop(self, env, data):
        traj, t = data
        beta = self.beta_estop

        reward_to_t = sum(env.compute_reward(s) for s, _ in traj[:t+1])
        full_reward = sum(env.compute_reward(s) for s, _ in traj)

        Z = logsumexp([beta * full_reward, beta * reward_to_t])
        return beta * reward_to_t - Z

    # (4) IMPROVEMENT: data = (improved_traj, original_traj)
    def _ll_improvement(self, env, data):
        beta = self.beta_improvement
        best_traj, orig_traj = data

        r_best = compute_reward_for_trajectory(env, best_traj)
        r_orig = compute_reward_for_trajectory(env, orig_traj)

        Z = logsumexp([beta * r_best, beta * r_orig])
        return beta * r_best - Z

    # ------------------------------------------------------------
    # MCMC core
    # ------------------------------------------------------------

    def generate_proposal(self, old, stdev, normalize=True):
        prop = old + stdev * np.random.randn(len(old))
        if normalize:
            n = np.linalg.norm(prop)
            if n > 0:
                prop = prop / n
        return prop

    def initial_solution(self):
        v = np.random.randn(self.num_mcmc_dims)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def run_mcmc(self, samples, stepsize, normalize=True, adaptive=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        T = int(samples)
        stdev = float(stepsize)
        accept_cnt = 0

        # Target acceptance rate
        target = 0.6
        horizon = max(1, T // 100)
        lr = 0.001
        hist = []

        # Allocate chain + likelihood arrays
        self.chain = np.zeros((T, self.num_mcmc_dims))
        self.likelihoods = np.zeros(T)

        # Initial point
        cur = self.initial_solution()
        cur_ll = self.calc_ll(cur)
        map_ll, map_sol = cur_ll, cur

        # --------------------------------------
        #              MCMC LOOP
        # --------------------------------------
        pbar = tqdm(range(T), desc="MCMC Sampling", ncols=80)

        for t in pbar:
            prop = self.generate_proposal(cur, stdev, normalize)
            prop_ll = self.calc_ll(prop)

            # Metropolis-Hastings acceptance
            accept = (prop_ll > cur_ll) or \
                    (np.random.rand() < np.exp(prop_ll - cur_ll))

            if accept:
                cur, cur_ll = prop, prop_ll
                accept_cnt += 1
                if cur_ll > map_ll:
                    map_ll, map_sol = cur_ll, cur

            self.chain[t] = cur
            self.likelihoods[t] = cur_ll

            # Update tqdm with useful info
            pbar.set_postfix({
                "LL": f"{cur_ll:.3f}",
                "acc_rate": f"{accept_cnt/(t+1):.3f}"
            })

            # Adaptive proposal step-size
            if adaptive:
                hist.append(1 if accept else 0)
                if (t + 1) % horizon == 0:
                    acc_rate = np.mean(hist[-horizon:])
                    stdev = max(
                        1e-5,
                        stdev + lr * (acc_rate - target) / np.sqrt(t + 1)
                    )

        self.accept_rate = accept_cnt / T
        self.map_sol = map_sol

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        b = int(len(self.chain) * burn_frac)
        return np.mean(self.chain[b::skip_rate], axis=0)