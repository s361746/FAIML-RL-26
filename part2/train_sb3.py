import argparse
import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from rand_wrapper import RandomizationWrapper


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


def main() -> None:
    args = parse_args()
    algo, strategy, env_type, timesteps = args.algo, args.sampling_strategy, args.env_type, args.timesteps  

    env = gym.make(
        "PandaPush-v3",
        reward_type="sparse",
    )

    env = RandomizationWrapper(
        env, 
        env_type=env_type,
        mass_range=(1.0, 5.0), 
        mode=strategy
    )
    
    env.reset(seed=args.seed)

    if algo.lower() == "ppo":
        model = PPO(
            "MultiInputPolicy", 
            env, 
            ent_coef=0.05, 
            batch_size=2048, 
            gamma=0.95,
            seed=args.seed,
            verbose=1
        )
    elif algo.lower() == "sac":
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )

        policy_kwargs = dict(
            net_arch=[256, 256, 256]
        )

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
            verbose=0,
        )

    # Train the agent
    print(f"Starting training for {algo.upper()} over {timesteps} timesteps...")
    start_time = time.time() 
    model.learn(total_timesteps=timesteps)
    end_time = time.time()  
    env.close()

    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTraining completed in: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} total seconds)\n")

    # Save the trained model
    save_name = f"{algo}_{strategy}_{env_type}_{timesteps // 1000}k"
    model.save(save_name)
    print(f"Model saved successfully as: {save_name}")


if __name__ == "__main__":
    main()