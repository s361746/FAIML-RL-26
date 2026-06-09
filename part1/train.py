import argparse
import gymnasium as gym
import numpy as np
import time
import random

import torch
from agent import Policy, Agent
from gymnasium.wrappers import RecordVideo

import matplotlib.pyplot as plt


"""
Parses command-line arguments for the model training script
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy on Hopper environment")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="reinforce",
        choices=["reinforce", "actor_critic"],
        help="Training algorithm mode",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Baseline value for REINFORCE (only used with --mode reinforce)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


"""
Main training, evaluation, and visualization pipeline for the Hopper-v5 environment
"""
def main():
    # Parse command-line arguments
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    env = gym.make('Hopper-v5')

    # Extract state and action space dimensionalities
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    # Choose the configuration from arguments
    mode, baseline = args.mode, args.baseline
    policy = Policy(state_space=state_dim, action_space=action_dim, mode=mode)
    agent = Agent(policy=policy, device='cpu', baseline=baseline)

    n_episodes = args.episodes
    tot_steps = 0
    best_reward = -float('inf')

    training_start_time = time.time()
    # Lists of performances for plotting
    history_total_rewards = []
    history_forward_rewards = []
    history_survive_rewards = []
    history_episode_lengths = []
    history_episode_times = []
    history_cumulative_times = []
    
    N_STEPS_HORIZON = 20

    # -------------------------------------------------------------------------
    # 2. Training Loop
    # -------------------------------------------------------------------------
    for episode in range(n_episodes):
        episode_start_time = time.time()
        state, _ = env.reset()
        done = False
        n_steps = 0

        # Metrics variables for this episode
        ep_reward = 0
        ep_steps = 0
        ep_forward = 0
        ep_survive = 0

        while not done:
            # Get action from the policy
            action, action_log_prob = agent.get_action(state, evaluation=False)
            action_numpy = action.detach().cpu().numpy()

            # Take a step in the environment
            new_state, reward, terminated, truncated, info = env.step(action_numpy)
            done = terminated or truncated

            # Track total rewards and environment metrics
            ep_reward += reward
            ep_steps += 1
            ep_forward += info.get('reward_forward', 0)
            ep_survive += info.get('reward_survive', 0)

            agent.store_outcome(state, new_state, action_log_prob, reward, done)

            # N-STEP UPDATE: Update the network every N steps or if the episode terminates
            if mode == "actor_critic" and (ep_steps % N_STEPS_HORIZON == 0 or done):
                agent.update_policy()

            state = new_state
            n_steps += 1

        # REINFORCE only updates at the end of a complete episode
        if mode == "reinforce":
            agent.update_policy()

        # Compute execution durations
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        cumulative_duration = episode_end_time - training_start_time

        tot_steps += n_steps

        # Save episode metrics to history for visualization
        history_total_rewards.append(ep_reward)
        history_episode_lengths.append(ep_steps)
        history_forward_rewards.append(ep_forward)
        history_survive_rewards.append(ep_survive)
        history_episode_times.append(episode_duration)
        history_cumulative_times.append(cumulative_duration)

        # Save model weights if agent sets a performance record
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), "videos/best_hopper_policy.pth")

        # Print training progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward_100 = np.mean(history_total_rewards[-100:])
            avg_time_100 = np.mean(history_episode_times[-100:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (Last 100): {avg_reward_100:.2f} | "
                  f"Avg Time/Ep: {avg_time_100:.3f}s | "
                  f"Total steps: {tot_steps}")

    # After the training loop ends
    final_returns = np.array(history_total_rewards)
    print(f"\n--- EXPERIMENT RESULTS ---")
    print(f"Mean Return: {final_returns.mean():.2f}")
    print(f"Std Return: {final_returns.std():.2f}")
    print(f"Total Compute Time: {history_cumulative_times[-1]:.2f} seconds")
    # -------------------------------------------------------------------------
    # 3. Video Rendering
    # -------------------------------------------------------------------------
    print(f"\nTraining finished! Loading best model (Record: {best_reward:.2f})...")
    agent.policy.load_state_dict(torch.load("videos/best_hopper_policy.pth"))
    
    # Set up environment and record the video file
    base_env = gym.make('Hopper-v5', render_mode='rgb_array')
    test_reward = 0
    render_env = RecordVideo(base_env, video_folder='videos', name_prefix='hopper_test_run', episode_trigger=lambda x: True)
        
    state, info = render_env.reset()
    done = False
    
    while not done:
        action, _ = agent.get_action(state, evaluation=True)
        action_numpy = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = render_env.step(action_numpy)
        done = terminated or truncated
        test_reward += reward

    # Close environments and save the file
    render_env.close()
    base_env.close()
    env.close()

    print(f"Final Test Run Reward: {test_reward:.2f}")
    print("Video saved successfully! Check the 'videos' folder.")
    print("Generating performance and temporal analysis charts...")

    # -------------------------------------------------------------------------
    # 4. Data Visualization
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    # Plot 1: Total reward and episode length
    ax1.plot(moving_average(history_total_rewards), label='Total Reward', color='blue')
    ax1.set_ylabel('Reward', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1_2 = ax1.twinx()
    ax1_2.plot(moving_average(history_episode_lengths), label='Episode Length', color='red', alpha=0.4)
    ax1_2.set_ylabel('Steps (Max 1000)', color='red')
    ax1_2.tick_params(axis='y', labelcolor='red')
    ax1.set_title('Learning Curve (Moving Average 50 ep)')

    # Plot 2: Reward breakdown (Survive vs. Forward)
    ax2.plot(moving_average(history_survive_rewards), label='Survive Bonus', color='green', linewidth=2)
    ax2.plot(moving_average(history_forward_rewards), label='Forward Bonus', color='purple', linewidth=2)
    ax2.set_title('Strategy Analysis: Reward Distribution')
    ax2.set_ylabel('Points')
    ax2.legend()

    # Plot 3: Computation time analysis
    ax3.plot(moving_average(history_episode_times), label='Time per Episode (Right)', color='brown', linestyle='--')
    ax3.set_ylabel('Seconds / Episode', color='brown')
    ax3.tick_params(axis='y', labelcolor='brown')

    ax3_2 = ax3.twinx()
    ax3_2.plot(history_cumulative_times, label='Total Cumulative Time (Left)', color='black', alpha=0.7)
    ax3_2.set_ylabel('Total Compute Time (Seconds)', color='black')
    ax3_2.tick_params(axis='y', labelcolor='black')
    ax3.set_title('Computational Resource Consumption Profile')
    ax3.set_xlabel('Episodes')

    plt.tight_layout()
    plt.savefig('videos/hopper_training_analysis.png')
    print("Charts saved to 'videos/hopper_training_analysis.png'. Execution complete!")


"""
Calculates the moving average of a 1D array or list using linear convolution.
"""
def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
if __name__ == '__main__':
    main()