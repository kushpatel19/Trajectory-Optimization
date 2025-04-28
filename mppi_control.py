import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

def get_cartpole_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the cartpole environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 1
    state_size = 6
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here
    hyperparams['lambda'] = 0.01
    hyperparams['Q'] = torch.eye(state_size)
    hyperparams['noise_sigma'] = torch.eye(action_size) * 10
    # hyperparams["Q"][:2,:2] = hyperparams["Q"][:2,:2] * 10
    hyperparams["Q"][0, 0] = 2.0     # cart x
    hyperparams["Q"][1, 1] = 5.0    # theta1
    hyperparams["Q"][2, 2] = 5.0    # theta2
    hyperparams["Q"][3, 3] = 0.01      # x_dot
    hyperparams["Q"][4, 4] = 0.1      # theta1_dot
    hyperparams["Q"][5, 5] = 0.1      # theta2_dot


    # ---
    return hyperparams


# class MPPIController(object):

#     def __init__(self, env, num_samples, horizon, hyperparams):
#         """

#         :param env: Simulation environment. Must have an action_space and a state_space.
#         :param num_samples: <int> Number of perturbed trajectories to sample
#         :param horizon: <int> Number of control steps into the future
#         :param hyperparams: <dic> containing the MPPI hyperparameters
#         """
#         self.env = env
#         self.T = horizon
#         self.K = num_samples
#         self.lambda_ = hyperparams['lambda']
#         self.action_size = env.action_space.shape[-1]
#         self.state_size = env.state_space.shape[-1]
#         self.goal_state = torch.zeros(self.state_size)  # This is just a container for later use
#         self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
#         # self.Q_terminal = torch.eye(self.state_size) * 1e4 # Terminal Cost Matrix (state_size, state_size)
#         self.Q_terminal = torch.diag(torch.tensor([10.0, 300.0, 1000.0, 1.0, 30.0, 30.0])) * 1e2

#         self.noise_mu = torch.zeros(self.action_size)
#         self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
#         self.noise_sigma_inv = torch.inverse(self.noise_sigma)
#         self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
#         self.u_init = torch.zeros(self.action_size)
#         self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

#     def reset(self):
#         """
#         Resets the nominal action sequence
#         :return:
#         """
#         self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

#     def command(self, state):
#         """
#         Run a MPPI step and return the optimal action.
#         :param state: torch tensor of shape (state_size,)
#         :return:
#         """
#         action = None
#         perturbations = self.noise_dist.sample((self.K, self.T))    # shape (K, T, action_size)
#         perturbed_actions = self.U + perturbations      # shape (K, T, action_size)
#         trajectory = self._rollout_dynamics(state, actions=perturbed_actions)
#         trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
#         self._nominal_trajectory_update(trajectory_cost, perturbations)
#         # select optimal action
#         action = self.U[0]
#         # final update nominal trajectory
#         self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
#         self.U[-1] = self.u_init # Initialize new end action
#         return action

#     def _rollout_dynamics(self, state_0, actions):
#         """
#         Roll out the environment dynamics from state_0 and taking the control actions given by actions
#         :param state_0: torch tensor of shape (state_size,)
#         :param actions: torch tensor of shape (K, T, action_size)
#         :return:
#          * trajectory: torch tensor of shape (K, T, state_size) containing the states along the trajectories given by
#                        starting at state_0 and taking actions.
#                        This tensor contains K trajectories of T length.
#          TIP 1: You may need to call the self._dynamics method.
#          TIP 2: At most you need only 1 for loop.
#         """
#         state = state_0.unsqueeze(0).repeat(self.K, 1) # transform it to (K, state_size)
#         trajectory = None
#         # --- Your code here
#         trajectory = torch.zeros((self.K, self.T, self.state_size))  # initialize trajectory tensor

#         for t in range(self.T):
#             state = self._dynamics(state, actions[:, t, :])
#             trajectory[:, t, :] = state
#         # ---
#         return trajectory

#     def _compute_trajectory_cost(self, trajectory, perturbations):
#         """
#         Compute the costs for the K different trajectories
#         :param trajectory: torch tensor of shape (K, T, state_size)
#         :param perturbations: torch tensor of shape (K, T, action_size)
#         :return:
#          - total_trajectory_cost: torch tensor of shape (K,) containing the total trajectory costs for the K trajectories
#         Observations:
#         * The trajectory cost be the sum of the state costs and action costs along the trajectories
#         * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
#         * Action costs should be given by (non_perturbed_action_i)^T noise_sigma^{-1} (perturbation_i)

#         TIP 1: the nominal actions (without perturbation) are stored in self.U
#         TIP 2: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references.
#         """
#         total_trajectory_cost = None
#         # --- Your code here

#         K,T,state_size = trajectory.shape
#         state_cost = (trajectory - self.goal_state) @ self.Q @ (trajectory - self.goal_state).permute(0,2,1)
#         state_cost = state_cost.diagonal(dim1=1,dim2=2).sum(dim=1)

#         delta_terminal = trajectory[:,-1,:] - self.goal_state  # last state
#         # terminal_cost = delta_terminal @ self.Q_terminal @ delta_terminal.permute(0,2,1)
#         terminal_cost = torch.sum(delta_terminal @ self.Q_terminal * delta_terminal, dim=1)

#         action_cost = self.U @ self.noise_sigma_inv @ perturbations.permute(0,2,1)
#         action_cost = self.lambda_ * action_cost.diagonal(dim1=1,dim2=2).sum(dim=1)

#         total_trajectory_cost = state_cost + action_cost + terminal_cost

#         # ---
#         return total_trajectory_cost

#     def _nominal_trajectory_update(self, trajectory_costs, perturbations):
#         """
#         Update the nominal action sequence (self.U) given the trajectory costs and perturbations
#         :param trajectory_costs: torch tensor of shape (K,)
#         :param perturbations: torch tensor of shape (K, T, action_size)
#         :return: No return, you just need to update self.U

#         TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
#         """
#         # --- Your code here
#         minimum = torch.min(trajectory_costs)
#         term = trajectory_costs - minimum
#         gamma = torch.exp((-1/self.lambda_)*term)
#         eta = torch.sum(gamma)
#         w_k = (1/eta) * gamma
#         self.U += (w_k.reshape(-1,1,1) * perturbations).sum(dim=0)

#         # ---

#     def _dynamics(self, state, action):
#         """
#         Query the environment dynamics to obtain the next_state in a batched format.
#         :param state: torch tensor of size (...., state_size)
#         :param action: torch tensor of size (..., action_size)
#         :return: next_state: torch tensor of size (..., state_size)
#         """
#         next_state = self.env.batched_dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
#         next_state = torch.tensor(next_state, dtype=state.dtype)
#         return next_state


# class MPPIController:
    
#     def __init__(self, env, num_samples, horizon, hyperparams):
#         """
#         Initializes the MPPI controller.

#         Args:
#             env: The environment with batched_dynamics(), action_space, and state_space.
#             num_samples: Number of control trajectory samples (K).
#             horizon: Planning horizon (T).
#             hyperparams: Dictionary with keys:
#                 - 'lambda': Scaling parameter for softmin weighting
#                 - 'Q': Quadratic state cost matrix (torch.Tensor)
#                 - 'noise_sigma': Covariance of control noise (torch.Tensor)
#         """
#         self.env = env
#         self.T = horizon
#         self.K = num_samples
#         self.lambda_ = hyperparams['lambda']
#         self.Q = hyperparams['Q']
#         self.device = self.Q.device

#         self.action_size = env.action_space.shape[-1]
#         self.state_size = env.state_space.shape[-1]

#         self.goal_state = torch.zeros(self.state_size, device=self.device)

#         # Cost matrices
#         # self.Q_terminal = torch.diag(torch.tensor([50.0, 30.0, 70.0, 0.01, 10.0, 10.0], device=self.device)) * 1e2
#         self.Q_terminal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device))

#         # Control and noise definitions
#         self.noise_mu = torch.zeros(self.action_size, device=self.device)
#         self.noise_sigma = hyperparams['noise_sigma'].to(self.device)
#         self.noise_sigma_inv = torch.inverse(self.noise_sigma)
#         self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

#         # Nominal trajectory
#         self.U = torch.zeros((self.T, self.action_size), device=self.device)
#         self.u_init = torch.zeros(self.action_size, device=self.device)

#         self.prev_cost = None
#         self.prev_error = None

#     def reset(self):
#         """Resets the nominal control sequence."""
#         self.U = torch.zeros((self.T, self.action_size), device=self.device)

#     def command(self, state):
#         """
#         Executes one MPPI step and returns the next control action.

#         Args:
#             state: Current state (torch tensor of shape [state_size,])

#         Returns:
#             action: Optimal control action (torch tensor of shape [action_size,])
#         """
#         error = torch.norm(state - self.goal_state)
#         if self.prev_error is not None and abs(self.prev_error - error) < 1e-4:
#             return torch.zeros_like(self.U[0])
#         self.prev_error = error

#         # Adaptive exploration
#         scale = torch.clamp(torch.norm(self.U) * 0.5, min=1.0, max=10.0)
#         self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma * scale)

#         perturbations = self.noise_dist.sample((self.K, self.T)).to(self.device)
#         perturbed_actions = self.U + perturbations

#         trajectory = self._rollout_dynamics(state, perturbed_actions)
#         trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
#         self._nominal_trajectory_update(trajectory_cost, perturbations)

#         action = self.U[0]
#         action = torch.clamp(action, -20.0, 20.0)
#         self.U = torch.roll(self.U, -1, dims=0)
#         self.U[-1] = self.u_init
#         return action

#     def _rollout_dynamics(self, state_0, actions):
#         """
#         Rolls out the dynamics for each sampled trajectory.

#         Args:
#             state_0: Initial state (torch tensor [state_size,])
#             actions: Control samples (K, T, action_size)

#         Returns:
#             trajectory: (K, T, state_size)
#         """
#         # state = state_0.unsqueeze(0).repeat(self.K, 1)
#         # trajectory = torch.zeros((self.K, self.T, self.state_size), device=self.device)

#         # for t in range(self.T):
#         #     state = self._dynamics(state, actions[:, t, :])
#         #     trajectory[:, t, :] = state

#         # return trajectory
#         state = state_0.unsqueeze(0).repeat(self.K, 1)
#         trajectory = []

#         for t in range(self.T):
#             action_t = actions[:, t, :]
#             state = self._dynamics(state, action_t)
#             trajectory.append(state.unsqueeze(1))

#         return torch.cat(trajectory, dim=1)  # (K, T, state_size)

#     def _compute_trajectory_cost(self, trajectory, perturbations):
#         """
#         Computes the total cost of each trajectory.

#         Args:
#             trajectory: (K, T, state_size)
#             perturbations: (K, T, action_size)

#         Returns:
#             total_trajectory_cost: (K,) tensor
#         """
#         K, T, _ = trajectory.shape
#         trajectory = trajectory.clone()

#         # Wrap angular components
#         trajectory[:, :, 1] = torch.atan2(torch.sin(trajectory[:, :, 1] - self.goal_state[1]),
#                                          torch.cos(trajectory[:, :, 1] - self.goal_state[1]))
#         trajectory[:, :, 2] = torch.atan2(torch.sin(trajectory[:, :, 2] - self.goal_state[2]),
#                                          torch.cos(trajectory[:, :, 2] - self.goal_state[2]))

#         delta_x = trajectory - self.goal_state
#         time_weights = torch.linspace(0.5, 1.5, steps=self.T, device=self.device)
#         state_cost = ((delta_x @ self.Q) * delta_x).sum(dim=2)
#         state_cost = (state_cost * time_weights).sum(dim=1)

#         delta_terminal = trajectory[:, -1, :] - self.goal_state
#         terminal_cost = torch.sum(delta_terminal @ self.Q_terminal * delta_terminal, dim=1)

#         action_cost = self.U @ self.noise_sigma_inv @ perturbations.transpose(1, 2)
#         action_cost = self.lambda_ * action_cost.diagonal(dim1=1, dim2=2).sum(dim=1)

#         total_trajectory_cost = state_cost + action_cost + terminal_cost
#         mean_cost = total_trajectory_cost.mean().item()
#         self.prev_cost = 0.9 * self.prev_cost + 0.1 * mean_cost if self.prev_cost else mean_cost
#         return total_trajectory_cost

#     def _nominal_trajectory_update(self, trajectory_costs, perturbations):
#         """
#         Updates the nominal control sequence using softmin weighting.

#         Args:
#             trajectory_costs: (K,) tensor
#             perturbations: (K, T, action_size)
#         """
#         beta = torch.min(trajectory_costs)
#         gamma = torch.exp(-(trajectory_costs - beta) / (self.lambda_ + 1e-6 + 0.1 * self.prev_cost))
#         eta = torch.sum(gamma)
#         weights = gamma / eta

#         weighted_perturb = (weights.view(-1, 1, 1) * perturbations).sum(dim=0)
#         best_k = torch.argmin(trajectory_costs)

#         self.U = 0.7 * self.U + 0.3 * (self.U + weighted_perturb)
#         self.U = 0.9 * self.U + 0.1 * perturbations[best_k]

#     def _dynamics(self, state, action):
#         """
#         Batched dynamics function call.

#         Args:
#             state: Tensor (..., state_size)
#             action: Tensor (..., action_size)

#         Returns:
#             next_state: Tensor (..., state_size)
#         """
#         next_state = self.env.batched_dynamics(state.cpu().numpy(), action.cpu().numpy())
#         # next_state = self.env.batched_dynamics_torch(state, action)
#         return torch.tensor(next_state, dtype=state.dtype, device=self.device)
    

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) controller for a 6‑dimensional
    double‑pendulum‑on‑cart system.

    Attributes:
        env: Simulation environment exposing batched_dynamics_torch().
        T (int): Planning horizon.
        K (int): Number of sampled trajectories.
        lambda_ (float): Softmin temperature.
        Q (Tensor[6×6]): State cost matrix.
        Q_terminal (Tensor[6×6]): Terminal state cost.
        noise_sigma (Tensor[1×1]): Control perturbation covariance.
        U (Tensor[T×1]): Nominal control sequence.
    """

    def __init__(self, env, num_samples, horizon, hyperparams):
        self.env = env
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']

        # State cost (local quadratic around upright) + we'll add swing‑up below
        self.Q = hyperparams['Q']
        self.device = self.Q.device

        # Uniform terminal cost (you can adjust per‑state weights here)
        self.Q_terminal = torch.eye(env.state_space.shape[-1], device=self.device) * 1e1

        # Control noise
        self.noise_mu = torch.zeros(env.action_space.shape[-1], device=self.device)
        self.noise_sigma = hyperparams['noise_sigma'].to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # Nominal sequence and initialization
        self.U = torch.zeros((self.T, env.action_space.shape[-1]), device=self.device)
        self.u_init = torch.zeros(env.action_space.shape[-1], device=self.device)

        # Upright goal
        self.goal_state = torch.zeros(env.state_space.shape[-1], device=self.device)

        # For diagnostics / smoothing
        self.prev_cost = None
        self.prev_error = None

    def reset(self):
        """Zero out the nominal action sequence."""
        self.U.zero_()

    def command(self, state):
        """
        Perform one MPPI iteration.

        Args:
            state (Tensor[6]): current state.

        Returns:
            action (Tensor[1]): first control in the updated sequence.
        """
        # simple convergence check
        err = torch.norm(state - self.goal_state)
        if self.prev_error is not None and abs(self.prev_error - err) < 1e-4:
            return torch.zeros_like(self.U[0])
        self.prev_error = err

        # adaptive noise scaling
        scale = torch.clamp(torch.norm(self.U), 1.0, 10.0)
        self.noise_dist = MultivariateNormal(self.noise_mu,
                                             covariance_matrix=self.noise_sigma * scale)

        # sample perturbations
        eps = self.noise_dist.sample((self.K, self.T)).to(self.device)  # (K,T,1)
        U_pert = self.U.unsqueeze(0) + eps                              # (K,T,1)

        # rollout K trajectories in parallel
        traj = self._rollout_dynamics(state, U_pert)                   # (K,T,6)
        costs = self._compute_trajectory_cost(traj, eps)               # (K,)

        # update nominal
        self._nominal_trajectory_update(costs, eps)

        # return first action
        u0 = self.U[0].clamp(-20.0, 20.0)
        # shift and refill
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        return u0

    def _rollout_dynamics(self, state0, U_batch):
        """
        Rolls out the dynamics for each sampled trajectory using the existing
        self._dynamics() (which wraps env.batched_dynamics).

        Args:
            state0:      Tensor of shape (state_size,), the initial state.
            U_batch:     Tensor of shape (K, T, action_size), the perturbed
                         control sequences for all K rollouts.

        Returns:
            trajectory:  Tensor of shape (K, T, state_size), the resulting
                         state sequences.
        """
        K, T, _ = U_batch.shape
        # start all K trajectories from the same initial state
        x = state0.unsqueeze(0).repeat(K, 1)       # (K, state_size)

        # collect K×T states
        traj = []
        for t in range(T):
            # apply the t-th batch of controls to each of the K states
            x = self._dynamics(x, U_batch[:, t, :])  # (K, state_size)
            traj.append(x.unsqueeze(1))               # (K,1,state_size)

        # concatenate into (K,T,state_size)
        return torch.cat(traj, dim=1)

    def _compute_trajectory_cost(self, traj, eps):
        """
        Cost for each of K trajectories:
          ∑ₜ [(xₜ - x*)ᵀQ(xₜ - x*) + swing_cost] + terminalᵀQ_term terminal
          + λ · ∑ₜ ∆uₜᵀ Σ⁻¹ ∆uₜ
        """
        K, T, D = traj.shape

        # wrap angles relative to goal (upright=0)
        θ1 = traj[:, :, 1]
        θ2 = traj[:, :, 2]
        wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))
        traj[:, :, 1] = wrap(θ1)
        traj[:, :, 2] = wrap(θ2)

        # state deviation
        dx = traj - self.goal_state

        # local quadratic cost
        state_cost = ((dx @ self.Q) * dx).sum(dim=2)  # (K,T)

        # add swing‑up cost: 1−cosθ for both links
        swing = (1 - torch.cos(traj[:, :, 1])) + (1 - torch.cos(traj[:, :, 2]))
        state_cost = state_cost + 100.0 * swing      # weight swing term

        # time weighting (optional)
        tw = torch.linspace(1.0, 1.0, T, device=self.device)

        # sum over time
        J_state = (state_cost * tw).sum(dim=1)       # (K,)

        # terminal cost
        xt = traj[:, -1, :] - self.goal_state
        J_term = (xt @ self.Q_terminal * xt).sum(dim=1)

        # control cost
        # eps: (K,T,1)  => action_cost: (K,)
        ctrl = (self.U @ self.noise_sigma_inv)  # (T,1) @ (1,1) = (T,1)
        J_ctrl = (ctrl.unsqueeze(0) * eps).sum(dim=(1,2))

        # total
        J = J_state + J_term + self.lambda_ * J_ctrl

        # running average for smoothing
        m = J.mean().item()
        self.prev_cost = 0.9 * self.prev_cost + 0.1 * m if self.prev_cost else m

        return J

    def _nominal_trajectory_update(self, J, eps):
        """
        Softmin update:
          wₖ ∝ exp(−(Jₖ − Jmin)/(λ+ε)) ,   U ← ∑ₖ wₖ (U + εₖ)
        """
        β = J.min()
        γ = torch.exp(-(J - β) / (self.lambda_ + 1e-6 + 0.1 * self.prev_cost))
        w = γ / γ.sum()

        Δ = (w.view(-1,1,1) * eps).sum(dim=0)
        best = torch.argmin(J)
        # blend in average + best
        self.U = 0.7 * self.U + 0.3 * (self.U + Δ)
        self.U = 0.9 * self.U + 0.1 * (self.U + eps[best])

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Query the environment’s numpy‐based batched_dynamics and return
        a torch tensor on the same device as `state`.
        Args:
            state:  (K, state_dim) torch.Tensor
            action: (K, action_dim) torch.Tensor
        Returns:
            next_state: (K, state_dim) torch.Tensor
        """
        # pull to CPU numpy, run pybullet sim, then push back to GPU
        ns = self.env.batched_dynamics(state.cpu().numpy(),
                                       action.cpu().numpy())
        return torch.tensor(ns, dtype=state.dtype, device=state.device)
    
