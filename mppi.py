import time
import torch
import imageio
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from cartpole_env import CartpoleEnv
from mppi_control import MPPIController, get_cartpole_mppi_hyperparams


def main():
    # Initialize environment
    env = CartpoleEnv(render=True)

    # Initialize initial and goal states
    init_state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]) + 0.01 * np.random.randn(6)
    goal_state = np.zeros(6)

    # Reset environment with initial state
    env.reset(init_state)

    # Initialize MPPI controller
    hyperparams = get_cartpole_mppi_hyperparams()
    controller = MPPIController(env, num_samples=800, horizon=20, hyperparams=hyperparams)
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    frames = []
    total_steps = 50

    start_time = time.time()

    state = env.get_state()

    for step in tqdm(range(total_steps), desc="Running MPPI Simulation"):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state_tensor)

        next_state = env.step(control.cpu().numpy())
        state = next_state.copy()

        # Capture frame for GIF
        width, height, rgbPixels, depthPixels, segPixels = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=env.view_matrix,
            projectionMatrix=env.proj_matrix
        )
        frame = np.reshape(rgbPixels, (240, 320, 4))[:, :, :3]
        frames.append(frame)

        # Optional: Slow down the simulation for better visualization (comment if not needed)
        time.sleep(0.02)

        # Early stopping if close to goal
        if np.linalg.norm(state[1:3] - goal_state[1:3]) < 1:
            print("Goal Reached!")
            break

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal Runtime: {total_runtime:.2f} seconds")

    # Save GIF
    print("Saving animation as GIF...")
    imageio.mimsave('mppi_simulation.gif', frames, fps=10)
    print("GIF saved as 'mppi_simulation.gif'")

    # Display the simulation using OpenCV
    print("\nPlaying the simulation...")
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("MPPI Simulation", frame_bgr)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
