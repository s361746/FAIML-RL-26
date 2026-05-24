
import gymnasium as gym
import numpy as np
import time

import torch
from agent import Policy, Agent
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

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
    tot_steps = 0

    # --- NUOVE LISTE PER I GRAFICI ---
    history_total_rewards = []
    history_episode_lengths = []
    history_forward_rewards = []
    history_survive_rewards = []
    history_ctrl_rewards = []

    best_reward = -float('inf')

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        n_steps = 0

        # Variabili temporanee per questo singolo episodio
        ep_reward = 0
        ep_steps = 0
        ep_forward = 0
        ep_survive = 0
        ep_ctrl = 0

        while not done:
            
            action, action_log_prob = agent.get_action(state, evaluation=False)
            action_numpy = action.detach().cpu().numpy()

            new_state, reward, terminated, truncated, info = env.step(action_numpy)
            done = terminated or truncated

            # --- ESTRAZIONE DATI PER I GRAFICI ---
            ep_reward += reward
            ep_steps += 1
            # info.get() evita errori se per caso la chiave non esiste in versioni vecchie
            ep_forward += info.get('reward_forward', 0)
            ep_survive += info.get('reward_survive', 0)
            ep_ctrl += info.get('reward_ctrl', 0)

            agent.store_outcome(state, new_state, action_log_prob, reward, done)

            state = new_state
            episode_reward += reward
            n_steps += 1

        # Se questo episodio ha battuto il record, salviamo il cervello del robot!
        if ep_reward > best_reward:
            best_reward = ep_reward
            # Salviamo lo stato della rete neurale in un file
            torch.save(agent.policy.state_dict(), "videos/best_hopper_policy.pth")
            # print(f"[*] Nuovo Record! Salvato modello con {best_reward:.2f} punti.")

        agent.update_policy()

        tot_steps += n_steps

        # ALLA FINE DELL'EPISODIO: Salviamo i totali nelle liste globali
        history_total_rewards.append(ep_reward)
        history_episode_lengths.append(ep_steps)
        history_forward_rewards.append(ep_forward)
        history_survive_rewards.append(ep_survive)
        history_ctrl_rewards.append(ep_ctrl)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | Total Reward: {episode_reward:.2f} | current steps: {tot_steps}")

    print(f"\nTraining finished! Caricamento del modello migliore (Record: {best_reward:.2f})...")
    # Carichiamo i pesi del record assoluto!
    agent.policy.load_state_dict(torch.load("videos/best_hopper_policy.pth"))
    
    # 1. Creiamo l'ambiente base
    base_env = gym.make('Hopper-v5', render_mode='rgb_array')
    # ... (il resto del tuo codice per il video) ...
    test_reward = 0

    # 2. Registrazione video SENZA il blocco 'with' (molto più sicuro per forzare il salvataggio)
    render_env = RecordVideo(base_env, video_folder='videos', name_prefix='hopper_test_run', episode_trigger=lambda x: True)
        
    state, info = render_env.reset()
    done = False
    
    while not done:
        action, _ = agent.get_action(state, evaluation=True)
        action_numpy = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = render_env.step(action_numpy)
        done = terminated or truncated
        test_reward += reward

    # 3. Chiusura ESPLICITA degli ambienti. Questo dice a ffmpeg: "Salva l'MP4 ORA!"
    render_env.close()
    base_env.close()
    env.close()

    print(f"Final Test Run Reward: {test_reward:.2f}")
    print("Video salvato con successo! Controlla la cartella 'videos'.")

    # --- INIZIO SEZIONE GRAFICI ---
    def moving_average(data, window_size=50):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    print("Generazione dei grafici in corso...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # GRAFICO 1: Reward Totale e Sopravvivenza
    ax1.plot(moving_average(history_total_rewards), label='Total Reward', color='blue')
    ax1.set_ylabel('Reward', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1_2 = ax1.twinx()
    ax1_2.plot(moving_average(history_episode_lengths), label='Episode Length', color='red', alpha=0.4)
    ax1_2.set_ylabel('Steps (Max 1000)', color='red')
    ax1_2.tick_params(axis='y', labelcolor='red')
    ax1.set_title('Curva di Apprendimento (Media Mobile 50 ep)')

    # GRAFICO 2: Scomposizione del Reward
    ax2.plot(moving_average(history_survive_rewards), label='Survive Bonus', color='green', linewidth=2)
    ax2.plot(moving_average(history_forward_rewards), label='Forward Bonus', color='purple', linewidth=2)
    ax2.plot(moving_average(history_ctrl_rewards), label='Control Cost', color='orange', linewidth=2)
    ax2.set_title('Analisi della Strategia: Da dove arrivano i punti?')
    ax2.set_xlabel('Episodi')
    ax2.set_ylabel('Punti')
    ax2.legend()

    plt.tight_layout()
    # Salva l'immagine
    plt.savefig('videos/hopper_training_analysis.png')
    
    # RIMOSSO plt.show() - In questo modo lo script si chiude da solo e rilascia tutte le risorse!
    print("Grafici salvati in 'videos/hopper_training_analysis.png'. Esecuzione completata!")



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