# Model Training Overview

This folder serves as the central hub for training Reinforcement Learning (RL) agents using **Stable Baselines3** (SB3) and **SB3 Contrib** on various Gymnasium environments.

## Project Structure

The codebase is organized by environment to keep training logic modular:

-   **`lunar_lander/`**: Contains scripts and models for the `LunarLanderContinuous-v3` environment. PPO is used here as a reliable baseline.
-   **`bipedal_walker/`**: Contains scripts and models for `BipedalWalkerHardcore-v3`. This includes both **PPO** (baseline) and **TQC** (advanced distributional RL), as the hardcore version requires more robust algorithms.
-   **`run_trained_agent.py`**: A **unified runner script**. Instead of writing separate evaluation scripts for every model, this script can load and visualize any trained agent by configuring the `model_info` dictionary.

## Prerequisites & Dependencies

To run the training or visualization scripts, you need the following Python packages:

-   **`gymnasium[box2d]`**: The environment suite (includes Box2D physics engine).
-   **`stable-baselines3`**: Core RL library implementation (PPO, etc.).
-   **`sb3-contrib`**: Additional algorithms (TQC) for advanced use cases.
-   **`shimmy`**: Compatibility layer for Gymnasium.

Install them via:
```bash
pip install gymnasium[box2d] stable-baselines3 sb3-contrib shimmy
```

## Shared Utilities

### `run_trained_agent.py`
This script abstracts away the model loading and environment creation. It supports:
-   **Multiple Algorithms**: PPO, TQC, and easily extensible to others (SAC, TD3).
-   **Unified Interface**: Consistent rendering and evaluation loop across different environments.
-   **Configuration**: Simple dictionary-based config to switch between models.
