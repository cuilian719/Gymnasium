import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("Starting improved BipedalWalker training...")
    
    # Configuration
    ENV_ID = "BipedalWalkerHardcore-v3"
    LOG_DIR = "tmp_ppo_bipedal_walker/"
    MODEL_PATH = "model_training/bipedal_walker/ppo_model"
    STATS_PATH = "model_training/bipedal_walker/ppo_vec_normalize.pkl"
    N_ENVS = 16  # Parallel environments
    TOTAL_TIMESTEPS = 5_000_000  # 5M steps should be enough to see walking
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Create Vectorized Environment with Normalization
    # We use make_vec_env for parallel execution
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, monitor_dir=LOG_DIR)
    
    # VecNormalize is crucial for BipedalWalker to handle different scales of observation/reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    print(f"Created {N_ENVS} parallel environments with VecNormalize.")

    # 2. Instantiate the agent with tuned hyperparameters
    # These settings are closer to what's used in RL Baselines3 Zoo for BipedalWalker
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01, # Slight exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR
    )

    # 3. Train the agent
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    
    print("Training finished (or interrupted).")

    # 4. Save Model and Normalization Stats
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Normalization stats saved to {STATS_PATH}")
    
    env.close()

    # --- Plotting Code ---
    print("Plotting training metrics...")
    try:
        # Load results from monitor files
        df = load_results(LOG_DIR)
        
        # Calculate rolling window for smoother plots
        window_size = 50
        df['rolling_reward'] = df['r'].rolling(window=window_size).mean()
        df['rolling_len'] = df['l'].rolling(window=window_size).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Reward Plot
        # We plot against total timesteps. 'l' is length, cumsum gives roughly total steps (for one env).
        # Since we have multiple envs, 't' (walltime) or index is often used.
        # But 'l'.cumsum() is a good approximation for x-axis if we want "timesteps" from a single stream perspective,
        # or we just use the index as "episodes completed" across all envs.
        # Let's use index as "Episodes".
        
        ax1.plot(df['r'], alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(df['rolling_reward'], color='blue', label=f'Rolling Mean ({window_size})')
        ax1.set_title("Training Reward")
        ax1.set_xlabel("Episodes")
        ax1.legend()
        
        # Length Plot
        ax2.plot(df['l'], alpha=0.3, color='orange', label='Episode Length')
        ax2.plot(df['rolling_len'], color='orange', label=f'Rolling Mean ({window_size})')
        ax2.set_title("Episode Length")
        ax2.set_xlabel("Episodes")
        ax2.legend()
        
        plt.tight_layout()
        plt.show() # This will pop up a window
        print("Plot displayed.")
        
    except Exception as e:
        print(f"Error plotting: {e}")
    # ---------------------

    # 5. Visualizing the trained agent
    # Note: We must load the normalization stats to evaluate/visualize properly
    print("\nVisualizing trained agent...")
    
    # Create a dummy environment for evaluation (single instance, render_mode='human')
    eval_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=dict(render_mode="human"))
    
    # Load the saved statistics
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    
    # IMPORTANT: Don't update stats during evaluation
    eval_env.training = False 
    eval_env.norm_reward = False # We want to see the raw reward
    
    # Load the agent
    loaded_model = PPO.load(MODEL_PATH)
    
    obs = eval_env.reset()
    try:
        for _ in range(2000):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            # Render is handled by the env creation with render_mode='human'
            
            if done:
                obs = eval_env.reset()
    except KeyboardInterrupt:
        print("Visualization interrupted.")
    finally:
        eval_env.close()
    
    print("Demonstration completed!")

if __name__ == "__main__":
    main()
