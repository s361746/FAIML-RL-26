import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import PPO, SAC
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="Algorithm to use for training (PPO or SAC)",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    algo, strategy, env_type, timesteps = args.algo, args.sampling_strategy, args.env_type, args.timesteps  

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        reward_type="dense",
    )

    # Wrap the environment using the RandomizationWrapper.
    env = RandomizationWrapper(env, env_type=env_type, mode=strategy)

    # Initialize the chosen algorithm model
    if algo.lower() == "ppo":
        # Use MultiInputPolicy because Panda-Gym outputs dictionary observation spaces
        model = PPO("MultiInputPolicy", env, ent_coef=0.05, batch_size=2048, verbose=1)
    elif algo.lower() == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1)

    # Train the agent
    print(f"Starting training for {algo.upper()} over {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    env.close()

    # Save the trained model
    save_name = f"{algo}+ent_coef_{strategy}_{env_type}_{timesteps // 1000}k"
    model.save(save_name)
    print(f"Model saved successfully as: {save_name}")


if __name__ == "__main__":
    main()