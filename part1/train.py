"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import numpy as np
import time
from agent import Policy, Agent

def main():
    
    env = gym.make('Hopper-v5')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    #TODO: implement training loop for REINFORCE and Actor-Critic using the agent defined in agent.py
    policy = Policy(state_space=state_dim, action_space=action_dim, mode=select_algorithm_mode())
    agent = Agent(policy=policy, device='cpu')

    n_episodes = 1000

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, action_log_prob = agent.get_action(state, evaluation=False)
            action_numpy = action.detach().cpu().numpy()

            new_state, reward, terminated, truncated, _ = env.step(action_numpy)
            done = terminated or truncated

            agent.store_outcome(state, new_state, action_log_prob, reward, done)

            state = new_state
            episode_reward += reward

        agent.update_policy()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | Total Reward: {episode_reward:.2f}")

    print("\nTraining finished! Firing up the renderer...")
    
    render_env = gym.make('Hopper-v5', render_mode='human')
    state, info = render_env.reset()
    done = False
    test_reward = 0

    for i in range(2000):
        action, _ = agent.get_action(state, evaluation=True)
        action_numpy = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = render_env.step(action_numpy)
        test_reward += reward
        time.sleep(0.1)

    print(f"Final Test Run Reward: {test_reward:.2f}")
    render_env.close()

    env.close()

def select_algorithm_mode():
    print("="*40)
    print("  SELEZIONE MODALITÀ DI ADDESTRAMENTO")
    print("="*40)
    print("1. REINFORCE (Task 2)")
    print("2. Actor-Critic (Task 3)")
    print("="*40)
    
    while True:
        scelta = input("Inserisci il numero della modalità desiderata (1 o 2): ").strip()
        
        if scelta == "1":
            return "reinforce"
        elif scelta == "2":
            return "actor_critic"
        else:
            print("Scelta non valida! Per favore, inserisci '1' per REINFORCE o '2' per Actor-Critic.\n")

if __name__ == '__main__':
    main()