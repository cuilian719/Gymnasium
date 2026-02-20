import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TQC
import os

def main():
    model_info = {
        "model_path": "model_training/lunar_lander/ppo_model",
        "env_id": "LunarLanderContinuous-v3",
        "algo": "PPO"
    }
    # model_info = {
    #     "model_path": "model_training/bipedal_walker/ppo_model",
    #     "env_id": "BipedalWalker-v3",
    #     "algo": "PPO"
    # }
    # model_info = {
    #     "model_path": "model_training/bipedal_walker/tqc_model",
    #     "env_id": "BipedalWalker-v3",
    #     "algo": "TQC"
    # }    
    
    if not os.path.exists(f"{model_info['model_path']}.zip"):
        print(f"Error: Model file '{model_info['model_path']}.zip' not found.")
        print("Please run reproduce_gym.py first to train and save the model.")
        return

    print(f"Loading model from {model_info['model_path']}...")
    if model_info['algo'] == "PPO":
        model = PPO.load(model_info['model_path'])
    elif model_info['algo'] == "TQC":
        model = TQC.load(model_info['model_path'])

    env_id = model_info['env_id']
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="human")

    print("Running trained agent...")
    obs, info = env.reset()
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
