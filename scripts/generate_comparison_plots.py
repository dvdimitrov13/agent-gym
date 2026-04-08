#!/usr/bin/env python3
"""Generate PNG comparison charts across V1, V2, V3 training runs."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(path):
    with open(path) as f:
        return json.load(f)["log_history"]


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def main():
    # Load all three runs
    v1 = load_metrics("checkpoints/qwen3-14b-grpo-v2/trainer_state_600.json")
    v2_all = load_metrics("checkpoints/qwen3-14b-grpo-v2/trainer_state_1200.json")
    v2 = [m for m in v2_all if m["step"] > 600]
    v3 = load_metrics("checkpoints/v3-cp600/trainer_state.json")

    # Normalize V2 steps to 1-600 for comparison
    v2_steps = np.array([m["step"] - 600 for m in v2])
    v1_steps = np.array([m["step"] for m in v1])
    v3_steps = np.array([m["step"] for m in v3])

    style = {"figure.facecolor": "white", "axes.facecolor": "white",
             "axes.grid": True, "grid.alpha": 0.3, "font.size": 11}
    plt.rcParams.update(style)

    colors = {"V1": "#2196F3", "V2": "#4CAF50", "V3": "#FF9800"}

    # --- Figure 1: Reward Trajectory ---
    fig, ax = plt.subplots(figsize=(10, 5))

    v1_reward = np.array([m["reward"] for m in v1])
    v2_reward = np.array([m["reward"] for m in v2])
    v3_reward = np.array([m["reward"] for m in v3])

    ax.plot(v1_steps, smooth(v1_reward), color=colors["V1"], linewidth=2,
            label="V1 (GRPO, thinking, additive)")
    ax.plot(v2_steps, smooth(v2_reward), color=colors["V2"], linewidth=2,
            label="V2 (GRPO, thinking, multiplicative)")
    ax.plot(v3_steps, smooth(v3_reward), color=colors["V3"], linewidth=2,
            label="V3 (DAPO, no thinking, multiplicative)")

    ax.set_xlabel("Step (normalized to 1-600)")
    ax.set_ylabel("Reward (MA-20)")
    ax.set_title("Reward Trajectory — All Training Runs")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 600)

    fig.tight_layout()
    fig.savefig("results/fig_reward_trajectory.png", dpi=150)
    print("Saved results/fig_reward_trajectory.png")
    plt.close()

    # --- Figure 2: Gradient Norms ---
    fig, ax = plt.subplots(figsize=(10, 5))

    v1_gnorm = np.array([m["grad_norm"] for m in v1])
    v2_gnorm = np.array([m["grad_norm"] for m in v2])
    v3_gnorm = np.array([m["grad_norm"] for m in v3])

    ax.plot(v1_steps, smooth(v1_gnorm, 10), color=colors["V1"], linewidth=2,
            label="V1 (GRPO, thinking, additive)")
    ax.plot(v2_steps, smooth(v2_gnorm, 10), color=colors["V2"], linewidth=2,
            label="V2 (GRPO, thinking, multiplicative)")
    ax.plot(v3_steps, smooth(v3_gnorm, 10), color=colors["V3"], linewidth=2,
            label="V3 (DAPO, no thinking, multiplicative)")

    ax.set_xlabel("Step (normalized to 1-600)")
    ax.set_ylabel("Gradient Norm (MA-10)")
    ax.set_title("Gradient Norms — All Training Runs")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 600)

    fig.tight_layout()
    fig.savefig("results/fig_gradient_norms.png", dpi=150)
    print("Saved results/fig_gradient_norms.png")
    plt.close()

    # --- Figure 3: Tool Usage ---
    fig, ax = plt.subplots(figsize=(10, 5))

    v1_tools = np.array([m["tools/call_frequency"] for m in v1])
    v2_tools = np.array([m["tools/call_frequency"] for m in v2])
    v3_tools = np.array([m["tools/call_frequency"] for m in v3])

    ax.plot(v1_steps, smooth(v1_tools), color=colors["V1"], linewidth=2,
            label="V1 (GRPO, thinking, additive)")
    ax.plot(v2_steps, smooth(v2_tools), color=colors["V2"], linewidth=2,
            label="V2 (GRPO, thinking, multiplicative)")
    ax.plot(v3_steps, smooth(v3_tools), color=colors["V3"], linewidth=2,
            label="V3 (DAPO, no thinking, multiplicative)")

    ax.set_xlabel("Step (normalized to 1-600)")
    ax.set_ylabel("Tool Calls per Rollout (MA-20)")
    ax.set_title("Tool Usage — All Training Runs")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 600)

    fig.tight_layout()
    fig.savefig("results/fig_tool_usage.png", dpi=150)
    print("Saved results/fig_tool_usage.png")
    plt.close()


if __name__ == "__main__":
    main()
