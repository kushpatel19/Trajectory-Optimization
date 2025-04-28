import time
import numpy as np
import pybullet as p
import imageio
import cv2
from tqdm import tqdm

from cartpole_env import CartpoleEnv
from ilqr_control import iLQRController


def main():
    # Initialize environment
    env = CartpoleEnv(render=True)

    # Initialize initial and goal states
    init_state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]) + 0.01 * np.random.randn(6)
    goal_state = np.zeros(6)

    env.reset(init_state)

    # Initialize iLQR controller
    ilqr = iLQRController(env, horizon=20, max_iters=50)

    frames = []
    total_steps = 50

    # Initialize action sequence U with random values
    U = np.random.uniform(-0.1, 0.1, (ilqr.horizon, ilqr.action_dim)) *0
    X = ilqr.rollout(init_state,U)

    start0_time = time.time()

    # state = env.get_state()

    for step in tqdm(range(total_steps), desc="Running iLQR Simulation"):
        # 1. Extract state from PyBullet
        state = env.get_state()

        # 2. Start time
        start_time = time.time()

        # 3. Run one step of iLQR (get new action)
        U, X = ilqr.command(state, U, X)
        action = U[0]

        # 4. End time
        end_time = time.time()

        # 5. Execute the first action
        next_state = env.step(action)
        state = next_state.copy()

        # 6. Render the frame for GIF
        width, height, rgbPixels, *_ = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=env.view_matrix,
            projectionMatrix=env.proj_matrix
        )
        frame = np.reshape(rgbPixels, (240, 320, 4))[:, :, :3]
        frames.append(frame)

        U[:-1] = U[1:]
        U[-1] = U[-2]
        X[:-1] = X[1:]
        X[-1] = X[-2]
        time.sleep(0.02)
        # 7. (Optional) Create actions for next pass (not needed since we recompute every step)

        # # 8. Set the description for tqdm progress bar
        # rollout_states = ilqr.rollout(state, np.tile(action, (ilqr.horizon, 1)))
        # rollout_cost = ilqr.cost(rollout_states, np.tile(action, (ilqr.horizon, 1)))
        # tqdm_desc = f"Iter {step:03d} | Cost: {rollout_cost:.1f} | Time: {(end_time - start_time):.3f} sec"
        # tqdm.write(tqdm_desc)  # to print nicely above tqdm bar

        # 9. Early stopping
        if np.linalg.norm(state[1:] - goal_state[1:]) < 0.5:
            print("Goal Reached!")
            break

    end0_time = time.time()
    print(f"\nTotal Runtime: {end0_time - start0_time:.2f} seconds")

    # Save GIF
    print("Saving animation as GIF...")
    imageio.mimsave('ilqr_simulation.gif', frames, fps=10)
    print("GIF saved as 'ilqr_simulation.gif'")

    # Display the simulation using OpenCV
    print("\nPlaying the simulation...")
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("iLQR Simulation", frame_bgr)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
