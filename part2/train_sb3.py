import os
import time
import torch
import argparse
import panda_gym
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from rand_wrapper import RandomizationWrapper


"""
Parses command-line arguments for the model training script
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC/PPO on PandaPush-v3")
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
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


"""
Main execution pipeline to train a RL model
"""
def main() -> None:
    # Parse command-line arguments
    args = parse_args()
    algo, strategy, env_type, timesteps = args.algo, args.sampling_strategy, args.env_type, args.timesteps  

    # Create directory for logging tensorboard data and model metrics
    log_dir = f"./logs/{algo}_{strategy}_{env_type}"
    os.makedirs(log_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    if algo.lower() == "ppo":
        
        # Initialize environment with dense rewards for on-policy training and normalization
        env = make_vec_env(
            "PandaPush-v3",
            seed=args.seed,
            env_kwargs={"reward_type": "dense"},
            wrapper_class=lambda e: RandomizationWrapper(e, env_type=env_type, mode=strategy)
        )
        
        # Create vectorized environment for parallelized/stable training
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        model = PPO(
            "MultiInputPolicy", 
            env, 
            learning_rate=3e-4,         # Step size for weight updates
            n_steps=2048,               # Steps per environment rollout
            batch_size=64,              # Mini-batch size for SGD
            n_epochs=10,                # Optimization passes per rollout
            ent_coef=0.01,              # Entropy bonus for exploration
            gamma=0.99,                 # Discount factor for future rewards
            seed=args.seed,             # Seed for reproducibility
            tensorboard_log=log_dir,    # Log directory for TensorBoard
            verbose=0                   # Verbosity level        
        )

    elif algo.lower() == "sac":
        
        # Initialize environment with sparse rewards for off-policy validation        
        env = make_vec_env(
            "PandaPush-v3",
            seed=args.seed,
            env_kwargs={"reward_type": "sparse"},
            wrapper_class=lambda e: RandomizationWrapper(e, env_type=env_type, mode=strategy)
        )

        # Configure HER memory settings
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
        policy_kwargs = dict(net_arch=[256, 256, 256])

        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,                         # Learning rate for networks
            batch_size=2048,                            # Replay buffer batch size
            gamma=0.95,                                 # Discount factor for rewards
            tau=0.05,                                   # Target network update rate
            train_freq=64,                              # Optimization interval steps
            gradient_steps=64,                          # Gradient steps per update
            learning_starts=10_000,                     # Steps before training starts
            replay_buffer_class=replay_buffer_class,    # Buffer class chosen for HER
            replay_buffer_kwargs=replay_buffer_kwargs,  # HER specific arguments
            policy_kwargs=policy_kwargs,                # Network layout parameters
            seed=args.seed,                             # Seed for reproducibility
            tensorboard_log=log_dir,                    # Log directory for TensorBoard
            verbose=0                                   # Verbosity level
        )

    if algo.lower() == "ppo":
        eval_env = make_vec_env(
            "PandaPush-v3",
            env_kwargs={"reward_type": "dense"},
            wrapper_class=lambda e: RandomizationWrapper(e, env_type=env_type, mode=strategy)
        )
        
        # Normalize evaluation observations while keeping test metrics unscaled
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    else:
        # Set up an unnormalized environment pipeline for SAC verification
        eval_env = make_vec_env(
            "PandaPush-v3",
            env_kwargs={"reward_type": "sparse"},
            wrapper_class=lambda e: RandomizationWrapper(e, env_type=env_type, mode=strategy)
    )

    # Set up periodic callback evaluation to save checkpoints automatically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # -------------------------------------------------------------------------
    # 2. Training Execution, Post-Processing & Serialization
    # -------------------------------------------------------------------------
    print(f"Starting training for {algo.upper()} over {timesteps} timesteps...")
    start_time = time.time() 
    
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    end_time = time.time()  

    if algo.lower() == "ppo":
        vec_norm_path = os.path.join(log_dir, "best_model", "vec_normalize.pkl")
        env.save(vec_norm_path)
        print(f"Statistiche VecNormalize salvate in: {vec_norm_path}")

    # Close active environments to release computational resources
    env.close()
    eval_env.close()

    # Compute execution durations
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTraining completed in: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} total seconds)\n")

    # Serialize model weights and parameters to disk
    save_name = f"{algo}_{strategy}_{env_type}_{timesteps // 1000}k"
    model.save(save_name)
    print(f"Final model saved successfully as: {save_name}")

if __name__ == "__main__":
    main()