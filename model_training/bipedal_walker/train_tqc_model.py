import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("Starting TQC BipedalWalker training...")
    
    # Configuration
    ENV_ID = "BipedalWalkerHardcore-v3"
    LOG_DIR = "tmp_tqc_bipedal_walker/"
    MODEL_PATH = "model_training/bipedal_walker/tqc_model"
    STATS_PATH = "model_training/bipedal_walker/tqc_vec_normalize.pkl"
    N_ENVS = 4 # TQC is off-policy, so fewer parallel envs are often better for wall-time efficiency vs sample efficiency trade-off
    TOTAL_TIMESTEPS = 2_000_000 
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Create Vectorized Environment
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, monitor_dir=LOG_DIR)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Instantiate the agent
    # Hyperparameters tuned for BipedalWalkerHardcore-v3
    # Based on RL Baselines3 Zoo
    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    
    model = TQC(
        "MlpPolicy",
        env,
        top_quantiles_to_drop_per_net=2, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        target_update_interval=1,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        use_sde=True, # State Dependent Exploration
        tensorboard_log=LOG_DIR
    )

    # 3. Train the agent
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    
    print("Training finished (or interrupted).")

    # 4. Save Model and Normalization Stats
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
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
        # We plot against total timesteps: 'l'.cumsum() is a good approximation for x-axis
        x_axis = df['l'].cumsum()
        
        ax1.plot(x_axis, df['r'], alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(x_axis, df['rolling_reward'], color='blue', label=f'Rolling Mean ({window_size})')
        ax1.set_title("Training Reward")
        ax1.set_xlabel("Timesteps")
        ax1.legend()
        
        # Length Plot
        ax2.plot(x_axis, df['l'], alpha=0.3, color='orange', label='Episode Length')
        ax2.plot(x_axis, df['rolling_len'], color='orange', label=f'Rolling Mean ({window_size})')
        ax2.set_title("Episode Length")
        ax2.set_xlabel("Timesteps")
        ax2.legend()
        
        plt.tight_layout()
        plt.show() # This will pop up a window
        print("Plot displayed.")
        
    except Exception as e:
        print(f"Error plotting: {e}")
    # ---------------------

    # 5. Visualizing the trained agent
    print("\nVisualizing trained agent...")
    
    # Create a dummy environment for evaluation (single instance, render_mode='human')
    eval_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=dict(render_mode="human"))
    
    # Load the saved statistics
    # Note: TQC training uses VecNormalize, so we must load it for evaluation too
    if os.path.exists(STATS_PATH):
        eval_env = VecNormalize.load(STATS_PATH, eval_env)
        eval_env.training = False 
        eval_env.norm_reward = False
    else:
        print("Warning: Normalization stats not found. Performance might be poor.")
    
    # Load the agent
    loaded_model = TQC.load(MODEL_PATH)
    
    obs = eval_env.reset()
    try:
        for _ in range(2000):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            if done:
                obs = eval_env.reset()
    except KeyboardInterrupt:
        print("Visualization interrupted.")
    finally:
        eval_env.close()
    
    print("Demonstration completed!")

if __name__ == "__main__":
    main()
