import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym
from pprint import pprint
import os

class CartpoleEnv(BaseEnv):

    def __init__(self, render= False, *args, **kwargs):
        self.cartpole = None
        self.render_enabled = render
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (6,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (4,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        p.resetSimulation()
        # assert os.path.exists("C:/Users/kush2/OneDrive/Documents/UMich/Sem2/Robot Learning/project/double_pendulum.urdf"), "URDF file not found!"
        # p.setAdditionalSearchPath(pd.getDataPath())
        # p.setAdditionalSearchPath("C:/Users/kush2/OneDrive/Documents/UMich/Sem2/Robot Learning/project")
        self.cartpole = p.loadURDF('cartpole.urdf')
        # self.cartpole = p.loadURDF('C:/Users/kush2/OneDrive/Documents/UMich/Sem2/Robot Learning/project/double_pendulum.urdf')
        # self.cartpole = np.array([int(0), int(0), int(0)])
        # self.cartpole = 0
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 2, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta1, theta1_dot = p.getJointState(self.cartpole, 1)[0:2]
        theta2, theta2_dot = p.getJointState(self.cartpole, 2)[0:2]
        theta2 = theta2 + theta1
        theta2_dot = theta2_dot + theta1_dot
        return np.array([x, theta1, theta2, x_dot, theta1_dot, theta2_dot])

    def set_state(self, state):
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        theta2 = theta2 - theta1
        theta2_dot = theta2_dot - theta1_dot
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta1, targetVelocity=theta1_dot)
        p.resetJointState(self.cartpole, 2, targetValue=theta2, targetVelocity=theta2_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30, shape=(1,))  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta1_lims = [-np.pi, np.pi]
        theta2_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta1_dot_lims = [-5 * np.pi, 5 * np.pi]
        theta2_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(
            low=np.array([x_lims[0], theta1_lims[0], theta2_lims[0], x_dot_lims[0], theta1_dot_lims[0], theta2_dot_lims[0]], dtype=np.float32),
            high=np.array([x_lims[1], theta1_lims[1], theta2_lims[1], x_dot_lims[1], theta1_dot_lims[1], theta2_dot_lims[1]],
                          dtype=np.float32))  # linear force # TODO: Verify that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 240
        self.render_w = 320
        base_pos = [0, 0, 0]
        cam_dist = 12
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (4,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (4, 4) representing Jacobian df/dx for dynamics f
            B: np.array of shape (4, 1) representing Jacobian df/du for dynamics f
        """
        A, B = None, None
        # --- Your code here

        A = np.zeros((6, 6))
        B = np.zeros((6, 1))

        for i in range(6):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            
            next_state_plus = self.dynamics(state_plus, control)
            next_state_minus = self.dynamics(state_minus, control)
            
            A[:, i] = (next_state_plus - next_state_minus) / (2 * eps)
            
            self.set_state(state)

        control_plus = control + eps
        control_minus = control - eps

        next_state_plus = self.dynamics(state, control_plus)
        next_state_minus = self.dynamics(state, control_minus)

        B[:, 0] = (next_state_plus - next_state_minus) / (2 * eps)


        # ---
        return A, B

    def batched_dynamics_torch(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor):
        """
        Safe GPU-friendly batched dynamics.

        Returns:
            next_state: torch.Tensor of shape (B, 6)
        """
        with torch.no_grad():
            next_state = dynamics_analytic(state_tensor, action_tensor)

        # Ensure correct type and shape
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=state_tensor.dtype)

        return next_state.to(state_tensor.device)


def linearize_numerical_fast(state, control, eps=1e-3):
    A = np.zeros((6, 6))
    B = np.zeros((6, 1))

    for i in range(6):
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[i] += eps
        state_minus[i] -= eps

        s_plus = torch.from_numpy(state_plus).float().unsqueeze(0)
        s_minus = torch.from_numpy(state_minus).float().unsqueeze(0)
        u = torch.from_numpy(control).float().unsqueeze(0)

        fx_plus = dynamics_analytic(s_plus, u).squeeze(0).detach().numpy()
        fx_minus = dynamics_analytic(s_minus, u).squeeze(0).detach().numpy()

        A[:, i] = (fx_plus - fx_minus) / (2 * eps)

    control_plus = control + eps
    control_minus = control - eps

    u_plus = torch.from_numpy(control_plus).float().unsqueeze(0)
    u_minus = torch.from_numpy(control_minus).float().unsqueeze(0)
    s = torch.from_numpy(state).float().unsqueeze(0)

    fu_plus = dynamics_analytic(s, u_plus).squeeze(0).detach().numpy()
    fu_minus = dynamics_analytic(s, u_minus).squeeze(0).detach().numpy()

    B[:, 0] = (fu_plus - fu_minus) / (2 * eps)

    return A, B


def dump_body_dynamics(body_uid: int):
    """
    Print/return {link_name: {mass, com, inertia_diag, length}} for every link
    (‑1 == base).  Works on any PyBullet build (no keyword args).
    """
    # from pprint import pprint
    # import numpy as np
    # import pybullet as p

    out = {}
    n_links = p.getNumJoints(body_uid)

    # pull the whole visual table once (fast) --------------
    visual_tbl = p.getVisualShapeData(body_uid)   # list of tuples
    # make a dict: link_index -> visual record
    visual_by_link = {rec[1]: rec for rec in visual_tbl}

    for link_idx in range(-1, n_links):
        name = ("base" if link_idx == -1
                else p.getJointInfo(body_uid, link_idx)[12].decode())

        # 1) mass / inertia / COM  --------------------------
        mass, _, inertia_diag, com_pos, com_orn, *_ = \
            p.getDynamicsInfo(body_uid, link_idx)

        # 2) quick length estimate from visual --------------
        length = None
        if link_idx in visual_by_link:
            shape_type = visual_by_link[link_idx][2]
            dims       = visual_by_link[link_idx][3]   # half‑extents etc.
            if shape_type == p.GEOM_BOX:               # [hx, hy, hz]
                length = 2 * dims[2]                   # Z‑side = pole
            elif shape_type == p.GEOM_CYLINDER:        # [radius, half‑len]
                length = dims[1] * 2

        out[name] = dict(
            mass          = mass,
            com_local_xyz = np.round(com_pos, 6).tolist(),
            inertia_diag  = np.round(inertia_diag, 8).tolist(),
            length        = length
        )

    # pprint(out)
    return out


# def dynamics_analytic(state, action):
#     """
#         Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
#         Should support batching
#     Args:
#         state: torch.tensor of shape (B, 4) representing the cartpole state
#         control: torch.tensor of shape (B, 1) representing the force to apply

#     Returns:
#         next_state: torch.tensor of shape (B, 4) representing the next cartpole state

#     """
#     next_state = None
#     dt = 0.05
#     g = 9.81
#     mc = 1
#     mp1 = 0.1
#     mp2 = 0.1
#     l1 = 0.5
#     l2 = 0.5

#     # --- Your code here
#     x, theta1, theta2, x_dot, theta1_dot, theta2_dot= torch.chunk(state, 6, dim=1)
#     force = action

#     # Precompute trigonometric terms
#     sin1 = torch.sin(theta1)
#     cos1 = torch.cos(theta1)
#     sin2 = torch.sin(theta2)
#     cos2 = torch.cos(theta2)
#     sin_diff = torch.sin(theta1 - theta2)
#     cos_diff = torch.cos(theta1 - theta2)

#     # Compute inertia matrix D (from paper equation 3)
#     d1 = mc + mp1 + mp2
#     d2 = (mp1/2 + mp2) * l1
#     d3 = mp2 * l2/2
#     d4 = (mp1/3 + mp2) * l1**2
#     d5 = mp2 * l1 * l2/2
#     d6 = mp2 * l2**2/3
    
#     D = torch.stack([
#         torch.cat([d1, d2*cos1, d3*cos2], dim=1),
#         torch.cat([d2*cos1, d4, d5*cos_diff], dim=1),
#         torch.cat([d3*cos2, d5*cos_diff, d6], dim=1)
#     ], dim=2)

#     # Compute Coriolis matrix C (corrected version)
#     C = torch.zeros_like(D)
#     C[..., 0, 1] = -d2*sin1*theta1_dot.squeeze(-1)
#     C[..., 0, 2] = -d3*sin2*theta2_dot.squeeze(-1)
#     C[..., 1, 0] = d2*sin1*theta1_dot.squeeze(-1)
#     C[..., 1, 2] = d5*sin_diff*theta2_dot.squeeze(-1)
#     C[..., 2, 0] = d3*sin2*theta2_dot.squeeze(-1)
#     C[..., 2, 1] = -d5*sin_diff*theta1_dot.squeeze(-1)

#     # Compute gravity vector G (from paper equation 5)
#     f1 = (mp1/2 + mp2) * l1 * g
#     f2 = mp2 * l2 * g/2
#     G = torch.stack([
#         torch.zeros_like(x),
#         -f1 * sin1,
#         -f2 * sin2
#     ], dim=2)

#     theta_dot = torch.cat([
#         x_dot,
#         theta1_dot,
#         theta2_dot
#     ], dim=-1).unsqueeze(-1)

#     # Compute accelerations (from paper equation 2)
#     theta_ddot = torch.linalg.solve(D, 
#         force.unsqueeze(-1) - C.matmul(theta_dot.unsqueeze(-1)) - G
#     )
    
#     # Extract accelerations
#     x_ddot = theta_ddot[..., 0, :]
#     theta1_ddot = theta_ddot[..., 1, :]
#     theta2_ddot = theta_ddot[..., 2, :]

#     # Euler integration
#     x_next = x + x_dot * dt
#     theta1_next = theta1 + theta1_dot * dt
#     theta2_next = theta2 + theta2_dot * dt
#     x_dot_next = x_dot + x_ddot * dt
#     theta1_dot_next = theta1_dot + theta1_ddot * dt
#     theta2_dot_next = theta2_dot + theta2_ddot * dt

#     next_state = torch.cat([
#         x_next,
#         theta1_next,
#         theta2_next,
#         x_dot_next,
#         theta1_dot_next,
#         theta2_dot_next
#     ], dim=-1)


#     # ---

#     return next_state

def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 6) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply
    Returns:
        next_state: torch.tensor of shape (B, 6) representing the next cartpole state
    """
    dt = 0.05
    g = 9.81

    # P = dump_body_dynamics(p.loadURDF('cartpole.urdf'))
    # mc = P['cart']['mass']
    # mp1 = P['pole1']['mass']
    # mp2 = P['pole2']['mass']
    # l1 = P['pole1']['length'] / 2
    # l2 = P['pole2']['length'] / 2
    mc = 1.0
    mp1 = 0.1
    mp2 = 0.1
    l1 = 1.0
    l2 = 1.0

    # Ensure consistent dtype and device
    dtype = state.dtype
    device = state.device

    # Split state into components
    x, theta1, theta2_rel, x_dot, theta1_dot, theta2_rel_dot = torch.chunk(state, 6, dim=1)
    theta2 = theta2_rel  # Convert relative to absolute for computation
    theta2_dot = theta2_rel_dot

    force = action.to(dtype=dtype).unsqueeze(-1)  # Ensure force is of shape (B, 1)

    # Precompute trigonometric terms
    sin1 = torch.sin(theta1)
    cos1 = torch.cos(theta1)
    sin2 = torch.sin(theta2)
    cos2 = torch.cos(theta2)
    sin_diff = torch.sin(theta1 - theta2)
    cos_diff = torch.cos(theta1 - theta2)

    # Compute inertia matrix D (batched version)
    batch_size = state.shape[0]
    
    # Compute coefficients with proper dtype
    d1 = torch.full((batch_size, 1), mc + mp1 + mp2, dtype=dtype, device=device)
    d2 = torch.full((batch_size, 1), (mp1/2 + mp2) * l1, dtype=dtype, device=device)
    d3 = torch.full((batch_size, 1), mp2 * l2/2, dtype=dtype, device=device)
    d4 = torch.full((batch_size, 1), (mp1/3 + mp2) * l1**2, dtype=dtype, device=device)
    d5 = torch.full((batch_size, 1), mp2 * l1 * l2/2, dtype=dtype, device=device)
    d6 = torch.full((batch_size, 1), mp2 * l2**2/3, dtype=dtype, device=device)
    
    # Construct batched D matrix (B, 3, 3)
    D = torch.stack([
        torch.cat([d1, d2*cos1, d3*cos2], dim=1),
        torch.cat([d2*cos1, d4, d5*cos_diff], dim=1),
        torch.cat([d3*cos2, d5*cos_diff, d6], dim=1)
    ], dim=2)

    # Compute Coriolis matrix C (batched version)
    C = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
    C[:, 0, 1] = (-d2*sin1*theta1_dot - d3*sin2*theta2_dot).squeeze(-1)
    C[:, 0, 2] = (-d3*sin2*theta2_dot).squeeze(-1)
    C[:, 1, 0] = (d2*sin1*theta1_dot).squeeze(-1)
    C[:, 1, 2] = (d5*sin_diff*theta2_dot).squeeze(-1)
    C[:, 2, 0] = (d3*sin2*theta2_dot).squeeze(-1)
    C[:, 2, 1] = (-d5*sin_diff*theta1_dot).squeeze(-1)

    # Compute gravity vector G (batched version)
    f1 = (mp1/2 + mp2) * l1 * g
    f2 = mp2 * l2 * g/2
    G = torch.stack([
        torch.zeros_like(x, dtype=dtype),
        -f1 * sin1,
        -f2 * sin2
    ], dim=-1).unsqueeze(-1)  # Shape: (B, 3, 1)

    # Compute velocities vector
    theta_dot = torch.cat([x_dot, theta1_dot, theta2_dot], dim=-1).unsqueeze(-1)  # Shape: (B, 3, 1)
    H = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 1)

    # Compute accelerations - ensure proper shapes
    # rhs = force.unsqueeze(-1) - torch.matmul(C, theta_dot) - G
    rhs = -torch.bmm(C, theta_dot) - G + H*force  # Shape: (B, 3, 1
    theta_ddot = torch.linalg.solve(D, rhs)  # Shape: (B, 3, 1)

    # Extract accelerations - ensure proper reshaping
    x_ddot = theta_ddot[..., 0, :]
    theta1_ddot = theta_ddot[..., 1, :]
    theta2_ddot = theta_ddot[..., 2, :]

    # Euler integration
    x_dot_next = (x_dot + x_ddot * dt).squeeze(-1)
    theta1_dot_next = (theta1_dot + theta1_ddot * dt).squeeze(-1)
    theta2_dot_next = (theta2_dot + theta2_ddot * dt).squeeze(-1)

    # --- CLIP angular velocities ---
    theta1_dot_next = torch.clamp(theta1_dot_next, -5*np.pi, 5*np.pi)
    theta2_dot_next = torch.clamp(theta2_dot_next, -5*np.pi, 5*np.pi)

    x_next = x + x_dot_next * dt
    theta1_next = theta1 + theta1_dot_next * dt
    theta2_next = theta2 + theta2_dot_next * dt
 
    # print ("Finally come to end")

    # # Print shapes before concatenation
    # print("\nShapes before concatenation:")
    # print(f"x_next shape: {x_next.shape} (should be [B, 1])")
    # print(f"theta1_next shape: {theta1_next.shape} (should be [B, 1])")
    # print(f"theta2_next shape: {theta2_next.shape} (should be [B, 1])")
    # print(f"x_dot_next shape: {x_dot_next.shape} (should be [B, 1])")
    # print(f"theta1_dot_next shape: {theta1_dot_next.shape} (should be [B, 1])")
    # print(f"theta2_dot_next shape: {theta2_dot_next.shape} (should be [B, 1])")
    
    # Verify all tensors have exactly 2 dimensions
    for name, tensor in [('x_next', x_next),
                        ('theta1_next', theta1_next),
                        ('theta2_next', theta2_next),
                        ('x_dot_next', x_dot_next),
                        ('theta1_dot_next', theta1_dot_next),
                        ('theta2_dot_next', theta2_dot_next)]:
        if len(tensor.shape) != 2:
            print(f"WARNING: {name} has {len(tensor.shape)} dimensions (shape: {tensor.shape})")

    try:
        theta2_rel_next = theta2_next 
        theta2_rel_dot_next = theta2_dot_next
        next_state = torch.cat([
            x_next,
            theta1_next,
            theta2_rel_next,
            x_dot_next,
            theta1_dot_next,
            theta2_rel_dot_next
        ], dim=1)
    except RuntimeError as e:
        print("\nERROR DURING CONCATENATION:")
        print(f"Error message: {str(e)}")
        print("Final tensor shapes:")
        for i, tensor in enumerate([x_next, theta1_next, theta2_next, 
                                  x_dot_next, theta1_dot_next, theta2_dot_next]):
            print(f"Tensor {i}: shape {tensor.shape}")
        raise

    return next_state

def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here
    state = state.reshape(1, 6)
    control = control.reshape(1, 1)
    # state.requires_grad = True
    # control.requires_grad = True

    jacobian = torch.autograd.functional.jacobian(dynamics_analytic, (state, control))

    A = jacobian[0].reshape(6, 6)
    B = jacobian[1].reshape(6, 1)


    # ---
    return A, B
