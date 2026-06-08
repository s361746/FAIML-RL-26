import argparse
import time
import os
import torch

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rand_wrapper import RandomizationWrapper

torch.set_num_threads(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC/PPO on PandaPush-v3")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Algorithm to use for training (PPO or SAC)")
    parser.add_argument("--sampling-strategy", type=str, default="none", choices=["none", "udr", "adr"], help="Sampling strategy for the object mass")
    parser.add_argument("--env-type", type=str, default="source", choices=["source", "target"], help="PandaPush environment type")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo, strategy, env_type, timesteps = args.algo, args.sampling_strategy, args.env_type, args.timesteps  

    log_dir = f"./logs/{algo}_{strategy}_{env_type}"
    os.makedirs(log_dir, exist_ok=True)


    if algo.lower() == "ppo":
        
        def make_env():
            e = gym.make("PandaPush-v3", reward_type="dense")
            e = RandomizationWrapper(e, env_type=env_type, mass_range=(1.0, 5.0), mode=strategy)
            e.reset(seed=args.seed)
            return Monitor(e)
            
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        
        model = PPO(
            "MultiInputPolicy", 
            env, 
            learning_rate=3e-4,     
            n_steps=2048,           
            batch_size=64,          
            n_epochs=10,            
            ent_coef=0.01,          
            gamma=0.99,             
            seed=args.seed,
            tensorboard_log=log_dir,
            verbose=0               
        )

    elif algo.lower() == "sac":
        
        env = gym.make("PandaPush-v3", reward_type="sparse")
        env = RandomizationWrapper(env, env_type=env_type, mass_range=(1.0, 5.0), mode=strategy)
        env = Monitor(env) 
        env.reset(seed=args.seed)

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
        policy_kwargs = dict(net_arch=[256, 256, 256])

        
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,
            batch_size=2048,
            gamma=0.95,
            tau=0.05,
            train_freq=64,
            gradient_steps=64,
            learning_starts=10_000,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            tensorboard_log=log_dir, 
            verbose=0,               
        )

    
    if algo.lower() == "ppo":
        def make_eval_env():
            e = gym.make("PandaPush-v3", reward_type="dense")
            e = RandomizationWrapper(e, env_type=env_type, mass_range=(1.0, 5.0), mode=strategy)
            return Monitor(e)
        eval_env = DummyVecEnv([make_eval_env])
        
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    else:
        
        eval_env_base = gym.make("PandaPush-v3", reward_type="sparse")
        eval_env_base = RandomizationWrapper(eval_env_base, env_type=env_type, mass_range=(1.0, 5.0), mode=strategy)
        eval_env = Monitor(eval_env_base)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Inizio training
    print(f"Starting training for {algo.upper()} over {timesteps} timesteps...")
    start_time = time.time() 
    
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    
    end_time = time.time()  

    if algo.lower() == "ppo":
        vec_norm_path = os.path.join(log_dir, "best_model", "vec_normalize.pkl")
        env.save(vec_norm_path)
        print(f"Statistiche VecNormalize salvate in: {vec_norm_path}")

    env.close()
    eval_env.close()

    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTraining completed in: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} total seconds)\n")

    save_name = f"{algo}_{strategy}_{env_type}_{timesteps // 1000}k"
    model.save(save_name)
    print(f"Final model saved successfully as: {save_name}")

if __name__ == "__main__":
    main()