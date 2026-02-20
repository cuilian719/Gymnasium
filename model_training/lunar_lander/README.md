# LunarLander-v3 Training with PPO

This folder contains the training script and trained model for the **LunarLanderContinuous-v3** environment using the Proximal Policy Optimization (PPO) algorithm.

## Files

- **`train_ppo_model.py`**: The Python script used to train the agent.
- **`ppo_model.zip`**: The saved trained model.
- **`train_ppo_model.png`**: Plot of the training metrics (Reward & Episode Length).

## Usage

### Training

To train the model from scratch (or continue training), run:

```bash
python3 model_training/lunar_lander/train_ppo_model.py
```

This will:
1. Initialize the `LunarLanderContinuous-v3` environment.
2. Train a PPO agent for 1,000,000 timesteps.
3. Save the model to `model_training/lunar_lander/ppo_model.zip`.
4. Generate a plot of the training progress (`train_ppo_model.png`).
5. Visualize the trained agent in a window.

### Running the Trained Agent

To see the agent in action, use the shared runner script in the `model_training` directory. 
Make sure `model_info` in `run_trained_agent.py` is configured for LunarLander:

```python
    model_info = {
        "model_path": "model_training/lunar_lander/ppo_model",
        "env_id": "LunarLanderContinuous-v3",
        "algo": "PPO"
    }
```

Then run:

```bash
python3 model_training/run_trained_agent.py
```

## Analysis of Results

### Training Metrics (`train_ppo_model.png`)
The training plot shows two key metrics over 1 million timesteps:

1.  **Episode Reward**:
    -   **Goal**: Solved is roughly 200 points.
    -   **Trend**: The reward should increase from ~ -200 (random) to positive values. A successful training run will show the reward consistently staying above 200 after convergence.
    -   **Convergence**: PPO typically solves LunarLanderContinuous within 300k-500k steps.

2.  **Episode Length**:
    -   **Trend**: Initially, episodes might be short (crashing) or long (hovering/drifting). As the agent learns to land efficiently, the episode length generally stabilizes.

### Model Performance
-   **Algorithm**: PPO (Proximal Policy Optimization)
-   **Policy**: `MlpPolicy` (Multi-Layer Perceptron)
-   **Hyperparameters**: Standard Stable Baselines3 defaults for `LunarLanderContinuous-v3` are usually sufficient, but tuning learning rate and batch size can improve convergence speed. PPO is robust and reliable for this environment.
