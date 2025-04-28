import control
import numpy as np
import scipy.linalg
import cvxpy as cp


class LinearMPC:

    def __init__(self, A, B, Q, R, horizon):
        self.dx = A.shape[0]  # state dimension (should be 6)
        self.du = B.shape[1]  # control dimension (should be 1)
        assert A.shape == (self.dx, self.dx)
        assert B.shape == (self.dx, self.du)
        self.H = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def compute_SM(self):
        """
        Computes the S and M matrices as defined in the ipython notebook

        All the variables you need should be class member variables already

        Returns:
            S: np.array of shape (horizon * dx, horizon * du) S matrix
            M: np.array of shape (horizon * dx, dx) M matrix

        """
        S, M = None, None

        # --- Your code here
        S = np.zeros((self.H * self.dx, self.H * self.du))
        M = np.zeros((self.H * self.dx, self.dx))

        Ap = np.eye(self.dx)
        for i in range(self.H):
          for j in range(self.H - i):
            S[self.dx * (i + j) : self.dx * (i + j + 1),
                  self.du * j       : self.du * (j + 1)] = Ap.dot(self.B)
          Ap = self.A.dot(Ap)

          M[self.dx * i : self.dx * (i + 1), :] = Ap
        # ---
        return S, M

    def compute_Qbar_and_Rbar(self):
        Q_repeat = [self.Q] * self.H
        R_repeat = [self.R] * self.H
        return scipy.linalg.block_diag(*Q_repeat), scipy.linalg.block_diag(*R_repeat)

    def compute_finite_horizon_lqr_gain(self):
        """
            Compute the controller gain G0 for the finite-horizon LQR

        Returns:
            G0: np.array of shape (du, dx)

        """
        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()

        G0 = None

        # --- Your code here
        SQS_R = np.dot(np.dot(S.T, Qbar), S) + Rbar
        # print(SQS_R.shape)
        SQM = np.dot(np.dot(S.T, Qbar), M)
        # print(SQM.shape)
        G0 = np.dot(-np.linalg.pinv(SQS_R), SQM)
        # print(G0.shape)
        G0 = G0[:self.du, :]  # Extract the first block for G0

        # ---

        return G0

    def compute_lqr_gain(self):
        """
            Compute controller gain G for infinite-horizon LQR
        Returns:
            Ginf: np.array of shape (du, dx)

        """
        Ginf = None
        theta_T_theta, _, _ = control.dare(self.A, self.B, self.Q, self.R)

        # --- Your code here
        BT_theta_B = np.dot(np.dot(self.B.T, theta_T_theta), self.B)
        BT_theta_A = np.dot(np.dot(self.B.T, theta_T_theta), self.A)
        Ginf = np.dot(-np.linalg.inv(self.R + BT_theta_B), BT_theta_A)

        # ---
        return Ginf

    def lqr_box_constraints_qp_shooting(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing with shooting

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls

        """

        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        U = None
        # --- Your code here
        U = cp.Variable((self.H*self.du,))

        # Define the objective function
        objective = cp.quad_form(S @ U + M @ x0, Qbar) + cp.quad_form(U, Rbar)
        
        # Define the constraints
        constraints = [U >= np.tile(u_min, self.H), U <= np.tile(u_max, self.H)]
        
        # Create and solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP)
        
        # Reshape the result to (horizon, du)
        U = U.value.reshape(self.H, self.du)

        # ---

        return U

    def lqr_box_constraints_qp_collocation(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing
            with collocation

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls
            X: np.array of shape (horizon, dx) containing sequence of optimal states

        """

        X, U = None, None

        # --- Your code here
        X = cp.Variable((self.H + 1, self.dx))
        U = cp.Variable((self.H, self.du))
        cost = 0
        constraint = []

        constraint.append(X[0,:] == x0)
        for t in range(0,self.H):
          cost += cp.quad_form(X[t+1,:],self.Q) + cp.quad_form(U[t],self.R)
          constraint.append(X[t+1,:] == self.A @ X[t,:] + self.B @ U[t])
          constraint.append(u_min <= U[t]) 
          constraint.append(u_max >= U[t])

        # Solve the quadratic program
        problem = cp.Problem(cp.Minimize(cost), constraint)
        problem.solve(solver=cp.SCS)

        # Extract control and state trajectories
        U = U.value
        X = X.value[1:]  # Exclude initial state
        
        # ---

        return U, X
