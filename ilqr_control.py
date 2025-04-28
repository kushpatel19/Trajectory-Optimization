import numpy as np

# ========== Dynamics ==========
def dynamics_analytic_numpy(state, action):
    dt = 0.05
    g = 9.81
    mc = 1.0
    mp1 = 0.1
    mp2 = 0.1
    l1 = 1.0
    l2 = 1.0

    x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
    force = action[0]

    sin1 = np.sin(theta1)
    cos1 = np.cos(theta1)
    sin2 = np.sin(theta2)
    cos2 = np.cos(theta2)
    sin_diff = np.sin(theta1 - theta2)

    d1 = mc + mp1 + mp2
    d2 = (mp1/2 + mp2) * l1
    d3 = mp2 * l2 / 2
    d4 = (mp1/3 + mp2) * l1**2
    d5 = mp2 * l1 * l2 / 2
    d6 = mp2 * l2**2 / 3

    D = np.array([
        [d1, d2 * cos1, d3 * cos2],
        [d2 * cos1, d4, d5 * np.cos(theta1 - theta2)],
        [d3 * cos2, d5 * np.cos(theta1 - theta2), d6]
    ]) + 1e-6 * np.eye(3)

    C = np.array([
        [0, -d2 * sin1 * theta1_dot - d3 * sin2 * theta2_dot, -d3 * sin2 * theta2_dot],
        [d2 * sin1 * theta1_dot, 0, d5 * sin_diff * theta2_dot],
        [d3 * sin2 * theta2_dot, -d5 * sin_diff * theta1_dot, 0]
    ])

    G = np.array([0, -(mp1/2 + mp2)*l1*g*np.sin(theta1), -mp2*l2*g/2*np.sin(theta2)])
    H = np.array([1, 0, 0])

    q_dot = np.array([x_dot, theta1_dot, theta2_dot])
    q_ddot = np.linalg.solve(D, -C @ q_dot - G + H * force)

    x_dot_next = x_dot + q_ddot[0] * dt
    theta1_dot_next = np.clip(theta1_dot + q_ddot[1] * dt, -5*np.pi, 5*np.pi)
    theta2_dot_next = np.clip(theta2_dot + q_ddot[2] * dt, -5*np.pi, 5*np.pi)

    x_next = x + x_dot_next * dt
    theta1_next = theta1 + theta1_dot_next * dt
    theta2_next = theta2 + theta2_dot_next * dt

    return np.array([x_next, theta1_next, theta2_next, x_dot_next, theta1_dot_next, theta2_dot_next])

class iLQRController:

    def __init__(self, env, horizon=50, max_iters=50):
        self.env = env
        self.horizon = horizon
        self.max_iters = max_iters

        self.state_dim = env.state_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.Q = np.diag([1, 100.0, 100.0, 1, 0.1, 0.1])
        self.Qf = np.diag([1, 1.0, 1.0, 1.0, 1.0, 1.0]) * 1000
        self.R = np.eye(self.action_dim) * 0.02

        self.mu = 20
        self.delta0 = 10

    def dynamics(self, state, action):
        return dynamics_analytic_numpy(state, action)

    def rollout(self, x0, U_seq):
        X = np.zeros((self.horizon+1, self.state_dim))
        X[0] = x0
        for t in range(self.horizon):
            X[t+1] = self.dynamics(X[t], U_seq[t])
        return X

    def cost(self, X, U):
        cost = 0
        for t in range(self.horizon):
            dx = X[t] - np.zeros(self.state_dim)
            cost += dx @ self.Q @ dx + U[t] @ self.R @ U[t]
        dx = X[-1] - np.zeros(self.state_dim)
        cost += dx @ self.Qf @ dx
        return cost

    def finite_difference(self, f, x, u, eps=1e-5):
        fx = np.zeros((self.state_dim, self.state_dim))
        fu = np.zeros((self.state_dim, self.action_dim))
        for i in range(self.state_dim):
            dx = np.zeros_like(x)
            dx[i] = eps
            fx[:, i] = (f(x + dx, u) - f(x - dx, u)) / (2*eps)
        for i in range(self.action_dim):
            du = np.zeros_like(u)
            du[i] = eps
            fu[:, i] = (f(x, u + du) - f(x, u - du)) / (2*eps)
        return fx, fu

    def command(self, x0, U_init, X_init):
        U = U_init.copy()
        X = X_init.copy()
        X[0] = x0

        for iteration in range(self.max_iters):
            fx_list, fu_list = [], []
            for t in range(self.horizon):
                fx, fu = self.finite_difference(self.dynamics, X[t], U[t], eps=1e-5)
                fx_list.append(fx)
                fu_list.append(fu)

            Vx = self.Qf @ (X[-1] - np.zeros(self.state_dim))
            Vxx = self.Qf.copy()

            k_seq = []
            K_seq = []
            diverged = False

            for t in reversed(range(self.horizon)):
                Qx = 2*self.Q @ X[t] + fx_list[t].T @ Vx        # (5a)
                Qu = 2*self.R @ U[t] + fu_list[t].T @ Vx                                     # (5b)

                Qxx = 2*self.Q + fx_list[t].T @ Vxx @ fx_list[t]    # (5c)
                Quu = 2*self.R + fu_list[t].T @ Vxx @ fu_list[t]    # (5d)
                Qux = fu_list[t].T @ Vxx @ fx_list[t]               # (5e)

                Quu_tilde = 2*self.R + fu_list[t].T @ (Vxx + self.mu * np.eye(self.action_dim)) @ fu_list[t]   # (10a)
                Qux_tilde = fu_list[t].T @ (Vxx + self.mu * np.eye(self.action_dim)) @ fx_list[t]              # (10b)

                # Quu_tilde = Quu + self.mu * np.eye(self.action_dim)  # (10a)
                # Qux_tilde = Qux                                     # (10b)

                try:
                    L = np.linalg.cholesky(Quu_tilde)
                    Quu_inv = np.linalg.inv(Quu_tilde)
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                k = -Quu_inv @ (Qu)           # (10c)
                K = -Quu_inv @ Qux_tilde    # (10d)

                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k       # (11b)
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K    # (11c)
                Vxx = 0.5 * (Vxx + Vxx.T)

                k_seq.insert(0, k)
                K_seq.insert(0, K)

            if diverged:
                self.mu *= self.delta0
                continue

            alpha = 1.0
            for line_search in range(10):
                X_new = np.zeros_like(X)
                U_new = np.zeros_like(U)
                X_new[0] = x0
                cost_new = 0

                for t in range(self.horizon):
                    du = alpha * k_seq[t] + K_seq[t] @ (X_new[t] - X[t])
                    U_new[t] = U[t] + du
                    X_new[t+1] = self.dynamics(X_new[t], U_new[t])

                cost_new = self.cost(X_new, U_new)
                if np.isnan(cost_new):
                    self.mu *= self.delta0
                    break

                if cost_new < self.cost(X, U):
                    X = X_new
                    U = U_new
                    self.mu = max(self.mu / self.delta0, 1e-6)
                    break
                else:
                    alpha *= 0.5
                    if alpha < 1e-4:
                        self.mu *= self.delta0
                        break
            else:
                self.mu *= self.delta0
            # print(f"  iLQR Iter {iteration:03d} | Cost: {self.cost(X, U):.2f}")

        return U, X

