# BipedalWalkerHardcore-v3 Training (PPO & TQC)

This folder contains training scripts and models for the **BipedalWalkerHardcore-v3** environment. We provide implementations for two algorithms: **PPO** (Proximal Policy Optimization) and **TQC** (Truncated Quantile Critics).

## Algorithms

### 1. PPO (Proximal Policy Optimization)
-   **Script**: `train_ppo_model.py`
-   **Model**: `ppo_model.zip`, `ppo_vec_normalize.pkl`
-   **Pros**: Stable, easier to tune.
-   **Cons**: May need more samples to solve Hardcore mode.

### 2. TQC (Truncated Quantile Critics)
-   **Script**: `train_tqc_model.py`
-   **Model**: `tqc_model.zip`, `tqc_vec_normalize.pkl`
-   **Pros**: Often more sample-efficient and robust for complex continuous control tasks like BipedalWalkerHardcore.
-   **Cons**: Slightly more complex implementation (requires `sb3-contrib`).

## Usage

### Training

To train the PPO model:
```bash
python3 model_training/bipedal_walker/train_ppo_model.py
```

To train the TQC model:
```bash
python3 model_training/bipedal_walker/train_tqc_model.py
```

Both scripts will save the model and normalization statistics (`vec_normalize.pkl`) upon completion or interruption.

### Running the Trained Agent

To run the trained agent, edit `model_training/run_trained_agent.py` and uncomment the desired model configuration:

**For PPO:**
```python
    model_info = {
        "model_path": "model_training/bipedal_walker/ppo_model",
        "env_id": "BipedalWalkerHardcore-v3", # Note: Hardcore-v3
        "algo": "PPO"
    }
```

**For TQC:**
```python
    model_info = {
        "model_path": "model_training/bipedal_walker/tqc_model",
        "env_id": "BipedalWalkerHardcore-v3",
        "algo": "TQC"
    }
```

Then run:
```bash
python3 model_training/run_trained_agent.py
```

## Analysis of Results

### Hardcore Mode Challenges
BipedalWalkerHardcore is significantly harder than the standard version due to obstacles (stumps, pitfalls).

1.  **Reward Threshold**: The environment is considered solved at a reward of **300**.
2.  **Training Curve**:
    -   **Initial Phase**: Agent struggles to even stand or walk (negative reward).
    -   **Walking Phase**: Learn to walk on flat terrain (reward ~0 to 100).
    -   **Hardcore Phase**: Learn to jump over obstacles and hurdles. This often requires millions of steps.

### PPO vs TQC
-   **PPO**: Good baseline. Should reach walking proficiency but might plateau before mastering all obstacles without extensive tuning or training time (5M+ steps).
-   **TQC**: Generally achieves higher rewards faster in this environment due to better handling of Q-value overestimation and distributional RL properties. It is the recommended algorithm for solving the Hardcore version efficiently.
