import os
import time
import argparse
import panda_gym
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import  VecNormalize 
from stable_baselines3.common.env_util import make_vec_env
from rand_wrapper import RandomizationWrapper


"""
Parses command-line arguments for the model evaluation script
"""
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
        help="Type of environment variant to evaluate on",
    )
    return parser.parse_args()


"""
Main execution pipeline to evaluate a trained RL model
"""
def main() -> None:
    # Parse command-line arguments
    args = parse_args()
    model_path, episodes, render, env_type = args.model_path, args.episodes, args.render, args.env_type

    if not os.path.exists(model_path): 
        raise FileNotFoundError(f"Model file not found: {model_path}")

    is_sparse = "her" in os.path.basename(model_path).lower()
    is_ppo = "ppo" in os.path.basename(model_path).lower()
    render_mode = "human" if render else "rgb_array"
        
    # Select dense rewards for PPO and SAC or sparse rewards for SAC+HER
    reward_type = "sparse" if is_sparse else "dense"

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------   
    env = make_vec_env(
        "PandaPush-v3",
        env_kwargs={"render_mode": render_mode, "reward_type": reward_type},
        wrapper_class=lambda e: RandomizationWrapper(e, env_type=env_type, mode="none")
    )
    
    # Load and configure observation/reward normalization for PPO framework
    if is_ppo:
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False     
            env.norm_reward = False  
        else:
            print(f"[WARNING] File VecNormalize non trovato in {vec_path}. PPO fallirà quasi sicuramente.")

    model = load_model(model_path, env=env)

    # -------------------------------------------------------------------------
    # 2. Evaluation Loop
    # -------------------------------------------------------------------------
    episode_returns = []
    successes = []
    
    for episode in range(1, episodes + 1):
       
        # Reset the environment to start a new episode
        obs = env.reset() 
        done = False
        episode_return = 0.0

        while not done:
            # Predict optimal action using the deterministic policy
            action, _ = model.predict(obs, deterministic=True)
           
            # Tracking returned metrics
            obs, rewards, dones, infos = env.step(action)
            episode_return += float(rewards[0])
            done = dones[0]
            
            # Real-time rendering speed
            if render:
                time.sleep(0.03)

        episode_returns.append(episode_return)
        info = infos[0]
        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    # -------------------------------------------------------------------------
    # 3. Performance Summary
    # -------------------------------------------------------------------------
    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    # Compute and report final task success rates if available
    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")
    
    print(successes)


"""
Dynamically loads a Stable-Baselines3 model based on its file name or path.
"""
def load_model(model_path: str, env: gym.Env):
    model_name = os.path.splitext(os.path.basename(model_path))[0].lower()

    # Instantiate the correct Stable-Baselines3 algorithm base class
    if "ppo" in model_name or "ppo" in model_path.lower():
        return PPO.load(model_path, env=env)
    if "sac" in model_name or "sac" in model_path.lower():
        return SAC.load(model_path, env=env)
        
    raise ValueError(f"Could not determine algorithm from filename: {model_name}")

if __name__ == "__main__":
    main()