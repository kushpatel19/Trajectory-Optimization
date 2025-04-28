import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import imageio
import torch
from tqdm import tqdm

from cartpole_env import CartpoleEnv
from mppi_control import MPPIController, get_cartpole_mppi_hyperparams
from ilqr_control import iLQRController

def run_controller(env_class, controller_class, controller_args, total_steps=100, save_gif_path=None):
    env = env_class(render=False)
    init_state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]) + 0.01 * np.random.randn(6)
    # init_state = np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0]) + 0.01 * np.random.randn(6)
    goal_state = np.zeros(6)
    env.reset(init_state)

    controller = controller_class(env, **controller_args)

    if controller_class == iLQRController:
        U = np.random.uniform(-2, 2, (controller.horizon, controller.action_dim)) * 0
        X = controller.rollout(init_state, U)

    frames = []
    costs = []
    state = env.get_state()

    pbar = tqdm(range(total_steps), desc=f"Running {controller_class.__name__}")

    for _ in pbar:
        if isinstance(controller, MPPIController):
            action = controller.command(torch.tensor(state, dtype=torch.float32))
        else:
            U, X = controller.command(state, U, X)
            action = U[0]

        next_state = env.step(action.cpu().numpy() if hasattr(action, 'cpu') else action)
        state = next_state.copy()

        if controller_class == iLQRController:
            U[:-1] = U[1:]
            U[-1] = U[-2]
            X[:-1] = X[1:]
            X[-1] = X[-2]

        # ==== Calculate Cost ====
        cost = np.linalg.norm(state[1:3])**2 + 0.1*np.linalg.norm(state[3:])**2  # Example cost
        costs.append(cost)

        # Capture frame
        width, height, rgbPixels, *_ = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=env.view_matrix,
            projectionMatrix=env.proj_matrix
        )
        frame = np.reshape(rgbPixels, (240, 320, 4))[:, :, :3]
        frames.append(frame)

        if np.linalg.norm(state[1:3] - goal_state[1:3]) < 0.1:
            print("Goal Reached:)")
            break

    # Save GIF if requested
    if save_gif_path is not None:
        print(f"Saving {save_gif_path}...")
        imageio.mimsave(save_gif_path, frames, fps=20)

    return frames, costs

def display_side_by_side(frames1, frames2, title1='MPPI', title2='iLQR', save_combined_gif = None):
    frames_len = min(len(frames1), len(frames2))
    combined_frames = []
    for i in range(frames_len):
        frame1 = frames1[i]
        frame2 = frames2[i]

        frame1 = cv2.resize(frame1, (320, 240))
        frame2 = cv2.resize(frame2, (320, 240))

        # Put titles
        frame1 = cv2.putText(frame1, title1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        frame2 = cv2.putText(frame2, title2, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        combined = np.hstack((cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR), 
                              cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)))
        combined_frames.append(combined)

        cv2.imshow('Comparison: MPPI vs iLQR', combined)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save the combined gif
    if save_combined_gif is not None:
        combined_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in combined_frames]
        print(f"Saving {save_combined_gif}...")
        imageio.mimsave(save_combined_gif, combined_rgb, fps=10)

def main():
    print("Running MPPI Simulation...")
    start_time = time.time()
    mppi_frames, mppi_costs = run_controller(
        env_class=CartpoleEnv,
        controller_class=MPPIController,
        controller_args={
            'num_samples': 500,
            'horizon': 20,
            'hyperparams': get_cartpole_mppi_hyperparams()
        },
        total_steps=50,
        save_gif_path='mppi.gif'
    )
    end_time = time.time()
    np.save('mppi_costs.npy', mppi_costs)
    print(f"MPPI Simulation completed in {end_time - start_time:.2f} seconds.\n")

    print("Running iLQR Simulation...")
    start_time = time.time()
    ilqr_frames, ilqr_costs = run_controller(
        env_class=CartpoleEnv,
        controller_class=iLQRController,
        controller_args={
            'horizon': 20,
            'max_iters': 100
        },
        total_steps=100,
        save_gif_path='ilqr.gif'
    )
    end_time = time.time()
    np.save('ilqr_costs.npy', ilqr_costs)
    print(f"iLQR Simulation completed in {end_time - start_time:.2f} seconds.\n")

    # print("Displaying simulations side by side...")
    display_side_by_side(mppi_frames, ilqr_frames, save_combined_gif="comparison.gif")

if __name__ == "__main__":
    main()
