import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
import panda_gym
import time

from rand_wrapper import RandomizationWrapper


def load_model(model_path: str, env: gym.Env):
    model_name = os.path.splitext(os.path.basename(model_path))[0].lower()

    if "ppo" in model_name:
        return PPO.load(model_path, env=env)
    if "sac" in model_name:
        return SAC.load(model_path, env=env)
        
    raise ValueError(f"Could not determine algorithm from filename: {model_name}")
    

def evaluate(model_path: str, n_episodes: int, render: bool, env_type: str) -> None:
    if not os.path.exists(model_path): 
        raise FileNotFoundError(f"Model file not found: {model_path}")

    render_mode = "human" if render else "rgb_array"
    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        reward_type="dense",
    )
    env = RandomizationWrapper(env, env_type=env_type, mode="none")
    
    model = load_model(model_path, env=env)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if render:
                time.sleep(0.03)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")
    
    print(successes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC/PPO models on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to an SB3 model zip file",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50, 
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, 
        default="target",
        choices=["source", "target"],
        help="Type of environment environment variant to evaluate on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        env_type=args.env_type,
    )
