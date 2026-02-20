import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("Testing Gymnasium installation with PPO...")
    
    # Configuration
    ENV_ID = "LunarLanderContinuous-v3"
    LOG_DIR = "tmp_ppo_lunar_lander/"
    MODEL_PATH = "model_training/lunar_lander/ppo_model"
    TOTAL_TIMESTEPS = 1_000_000

    # 0. Setup log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Create environment
    # Wrap the environment with Monitor to log data
    env = Monitor(gym.make(ENV_ID), LOG_DIR)
    print(f"Created environment: {env}")

    # 2. Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("Training finished.")

    # 4. Save and reload (optional, but good practice)
    model.save(MODEL_PATH)
    del model # remove to demonstrate loading
    model = PPO.load(MODEL_PATH)
    
    # --- Plotting Code ---
    print("Plotting training metrics...")
    try:
        # Monitor logs are CSV files with a header line
        df = pd.read_csv(os.path.join(LOG_DIR, "monitor.csv"), skiprows=1)
        
        # Calculate rolling window for smoother plots
        window_size = 50
        df['rolling_reward'] = df['r'].rolling(window=window_size).mean()
        df['rolling_len'] = df['l'].rolling(window=window_size).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Reward Plot
        ax1.plot(df['r'], alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(df['rolling_reward'], color='blue', label=f'Rolling Mean ({window_size})')
        ax1.set_title("Training Reward")
        ax1.legend()
        
        # Length Plot
        ax2.plot(df['l'], alpha=0.3, color='orange', label='Episode Length')
        ax2.plot(df['rolling_len'], color='orange', label=f'Rolling Mean ({window_size})')
        ax2.set_title("Episode Length")
        ax2.legend()
        
        plt.tight_layout()
        plt.show() # This will pop up a window
        print("Plot displayed.")
        
    except Exception as e:
        print(f"Error plotting: {e}")
    # ---------------------

    # 5. Visualize the trained agent
    print("Visualizing trained agent...")
    env_eval = gym.make(ENV_ID, render_mode="human")
    
    obs, info = env_eval.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        
        if terminated or truncated:
            obs, info = env_eval.reset()

    env_eval.close()
    print("Demonstration completed!")

if __name__ == "__main__":
    main()
