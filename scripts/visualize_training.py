#!/usr/bin/env python3
"""Visualize GRPO training metrics from trainer_state.json.

Produces a multi-panel HTML dashboard with interactive Plotly charts.
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

# Use plotly for interactive HTML charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("Installing plotly...")
    os.system(f"{sys.executable} -m pip install plotly kaleido")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["log_history"]


def smooth(values, window=10):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Pad to avoid shrinking
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def parse_log_judge_scores(log_path):
    """Extract per-step LLM judge scores from train log."""
    import re
    scores = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"LLM judge: (\d+)/(\d+) scored, avg=([0-9.]+)", line)
            if m:
                scored, total, avg = int(m.group(1)), int(m.group(2)), float(m.group(3))
                scores.append({"scored": scored, "total": total, "avg": avg,
                               "submit_rate": scored / total})
    return scores


def parse_log_tool_calls(log_path):
    """Extract tool call patterns from train log."""
    import re
    patterns = []
    current = {"searches": [], "reads": [], "submits": []}
    for line in open(log_path):
        m = re.search(r"TI/TO \[(\d+)\] (search|read|submit_answer)\((.+?)\)", line)
        if m:
            idx, tool, args = m.group(1), m.group(2), m.group(3)
            if tool == "search":
                current["searches"].append(args)
            elif tool == "read":
                current["reads"].append(args)
            elif tool == "submit_answer":
                current["submits"].append(args)
        m2 = re.search(r"TI/TO done: (\d+) submitted, (\d+) no-tool, (\d+) still active, (\d+) tool calls", line)
        if m2:
            current["submitted"] = int(m2.group(1))
            current["no_tool"] = int(m2.group(2))
            current["still_active"] = int(m2.group(3))
            current["total_tools"] = int(m2.group(4))
            patterns.append(current)
            current = {"searches": [], "reads": [], "submits": []}
    return patterns


def build_dashboard(metrics, log_path, output_path):
    steps = [m["step"] for m in metrics]

    # Curriculum boundaries
    curriculum_lines = [
        (30, "2-hop start"),
        (70, "3-hop start"),
        (200, "Shuffled epoch 1"),
        (400, "Shuffled epoch 2"),
        # Could add more based on dataset structure
    ]

    # === FIGURE 1: Main reward dashboard (2x2) ===
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Reward (smoothed + raw)",
            "LLM Judge Score",
            "Efficiency & Format Rewards",
            "Reward Std (exploration signal)",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 1a: Total reward
    rewards = np.array([m["reward"] for m in metrics])
    fig1.add_trace(go.Scatter(x=steps, y=rewards, mode="markers", name="Raw reward",
                              marker=dict(size=2, color="lightblue", opacity=0.4)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=steps, y=smooth(rewards, 20), mode="lines", name="Reward (MA-20)",
                              line=dict(color="blue", width=2)), row=1, col=1)

    # 1b: Judge score
    judge = np.array([m["rewards/llm_judge_reward/mean"] for m in metrics])
    fig1.add_trace(go.Scatter(x=steps, y=judge, mode="markers", name="Raw judge",
                              marker=dict(size=2, color="lightsalmon", opacity=0.4)), row=1, col=2)
    fig1.add_trace(go.Scatter(x=steps, y=smooth(judge, 20), mode="lines", name="Judge (MA-20)",
                              line=dict(color="red", width=2)), row=1, col=2)

    # 1c: Efficiency + Format
    eff = np.array([m["rewards/efficiency_reward/mean"] for m in metrics])
    fmt = np.array([m["rewards/format_reward/mean"] for m in metrics])
    fig1.add_trace(go.Scatter(x=steps, y=smooth(eff, 20), mode="lines", name="Efficiency (MA-20)",
                              line=dict(color="green", width=2)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=steps, y=smooth(fmt, 20), mode="lines", name="Format (MA-20)",
                              line=dict(color="orange", width=2)), row=2, col=1)

    # 1d: Reward std
    rstd = np.array([m["reward_std"] for m in metrics])
    fig1.add_trace(go.Scatter(x=steps, y=smooth(rstd, 20), mode="lines", name="Reward Std (MA-20)",
                              line=dict(color="purple", width=2)), row=2, col=2)
    zero_std = np.array([m.get("frac_reward_zero_std", 0) for m in metrics])
    fig1.add_trace(go.Scatter(x=steps, y=smooth(zero_std, 20), mode="lines", name="Zero-std frac (MA-20)",
                              line=dict(color="gray", width=1, dash="dash")), row=2, col=2)

    # Add curriculum boundaries to all subplots
    for step_val, label in curriculum_lines:
        for r in [1, 2]:
            for c in [1, 2]:
                fig1.add_vline(x=step_val, line_dash="dot", line_color="gray",
                               opacity=0.5, row=r, col=c)

    fig1.update_layout(height=700, width=1100, title_text="GRPO Training — Reward Dashboard (600 steps)",
                       showlegend=True, template="plotly_white",
                       legend=dict(font=dict(size=9)))

    # === FIGURE 2: Training dynamics (2x2) ===
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Completion Length (mean, min, max)",
            "Tool Call Frequency",
            "Grad Norm & Entropy",
            "Step Time (seconds)",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 2a: Completion length
    mean_len = np.array([m["completions/mean_length"] for m in metrics])
    min_len = np.array([m["completions/min_length"] for m in metrics])
    max_len = np.array([m["completions/max_length"] for m in metrics])
    fig2.add_trace(go.Scatter(x=steps, y=smooth(mean_len, 10), mode="lines", name="Mean length",
                              line=dict(color="blue", width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=steps, y=smooth(min_len, 10), mode="lines", name="Min length",
                              line=dict(color="lightblue", width=1)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=steps, y=smooth(max_len, 10), mode="lines", name="Max length",
                              line=dict(color="darkblue", width=1)), row=1, col=1)

    # 2b: Tool calls
    tools = np.array([m["tools/call_frequency"] for m in metrics])
    fig2.add_trace(go.Scatter(x=steps, y=tools, mode="markers", name="Tool calls/rollout",
                              marker=dict(size=2, color="lightgreen", opacity=0.4)), row=1, col=2)
    fig2.add_trace(go.Scatter(x=steps, y=smooth(tools, 20), mode="lines", name="Tools (MA-20)",
                              line=dict(color="green", width=2)), row=1, col=2)

    # 2c: Grad norm + entropy
    gnorm = np.array([m["grad_norm"] for m in metrics])
    entropy = np.array([m["entropy"] for m in metrics])
    fig2.add_trace(go.Scatter(x=steps, y=smooth(gnorm, 10), mode="lines", name="Grad norm (MA-10)",
                              line=dict(color="red", width=2)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=steps, y=smooth(entropy, 10), mode="lines", name="Entropy (MA-10)",
                              line=dict(color="orange", width=2)), row=2, col=1)

    # 2d: Step time
    stime = np.array([m["step_time"] for m in metrics])
    fig2.add_trace(go.Scatter(x=steps, y=stime, mode="markers", name="Step time",
                              marker=dict(size=2, color="lightcoral", opacity=0.4)), row=2, col=2)
    fig2.add_trace(go.Scatter(x=steps, y=smooth(stime, 10), mode="lines", name="Step time (MA-10)",
                              line=dict(color="brown", width=2)), row=2, col=2)

    for step_val, label in curriculum_lines:
        for r in [1, 2]:
            for c in [1, 2]:
                fig2.add_vline(x=step_val, line_dash="dot", line_color="gray",
                               opacity=0.5, row=r, col=c)

    fig2.update_layout(height=700, width=1100, title_text="GRPO Training — Dynamics Dashboard (600 steps)",
                       showlegend=True, template="plotly_white",
                       legend=dict(font=dict(size=9)))

    # === FIGURE 3: Reward hacking detection ===
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Judge vs Format Reward (reward hacking check)",
            "Judge Score Distribution (early vs mid vs late)",
            "Completion Length vs Judge Score",
            "Submit Rate & No-tool Rate from Log",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 3a: Judge vs Format — if format is high but judge is low, that's reward hacking
    fig3.add_trace(go.Scatter(x=fmt, y=judge, mode="markers", name="Format vs Judge",
                              marker=dict(size=4, color=np.array(steps), colorscale="Viridis",
                                          showscale=True, colorbar=dict(title="Step", len=0.4, y=0.8)),
                              text=[f"Step {s}" for s in steps]), row=1, col=1)
    fig3.update_xaxes(title_text="Format Reward", row=1, col=1)
    fig3.update_yaxes(title_text="Judge Score", row=1, col=1)

    # 3b: Judge distribution across training phases
    early = judge[:100]  # steps 1-100
    mid = judge[200:400]  # steps 200-400
    late = judge[450:600]  # steps 450-600
    fig3.add_trace(go.Histogram(x=early, name="Steps 1-100", opacity=0.6,
                                marker_color="red", nbinsx=20), row=1, col=2)
    fig3.add_trace(go.Histogram(x=mid, name="Steps 200-400", opacity=0.6,
                                marker_color="blue", nbinsx=20), row=1, col=2)
    fig3.add_trace(go.Histogram(x=late, name="Steps 450-600", opacity=0.6,
                                marker_color="green", nbinsx=20), row=1, col=2)
    fig3.update_layout(barmode="overlay")

    # 3c: Completion length vs judge score
    fig3.add_trace(go.Scatter(x=mean_len, y=judge, mode="markers", name="Length vs Judge",
                              marker=dict(size=4, color=np.array(steps), colorscale="Viridis",
                                          showscale=False),
                              text=[f"Step {s}" for s in steps]), row=2, col=1)
    fig3.update_xaxes(title_text="Mean Completion Length", row=2, col=1)
    fig3.update_yaxes(title_text="Judge Score", row=2, col=1)

    # 3d: Submit patterns from log
    log_judges = parse_log_judge_scores(log_path)
    tito_patterns = parse_log_tool_calls(log_path)
    if tito_patterns:
        submit_rates = [p["submitted"] / max(p["submitted"] + p["no_tool"] + p["still_active"], 1)
                       for p in tito_patterns]
        no_tool_rates = [p["no_tool"] / max(p["submitted"] + p["no_tool"] + p["still_active"], 1)
                        for p in tito_patterns]
        pat_steps = list(range(len(submit_rates)))
        fig3.add_trace(go.Scatter(x=pat_steps, y=smooth(np.array(submit_rates), 5), mode="lines",
                                  name="Submit rate", line=dict(color="green", width=2)), row=2, col=2)
        fig3.add_trace(go.Scatter(x=pat_steps, y=smooth(np.array(no_tool_rates), 5), mode="lines",
                                  name="No-tool rate", line=dict(color="red", width=2)), row=2, col=2)

    fig3.update_layout(height=700, width=1100, title_text="GRPO Training — Reward Hacking Analysis",
                       showlegend=True, template="plotly_white",
                       legend=dict(font=dict(size=9)))

    # === FIGURE 4: Curriculum analysis ===
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Judge Score by Curriculum Phase",
            "Tool Calls by Curriculum Phase",
            "Efficiency by Curriculum Phase",
            "Clipped Ratio (truncation rate)",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # Phase boundaries
    phases = {
        "1-hop (0-29)": (0, 30),
        "2-hop (30-69)": (30, 70),
        "3-hop (70-199)": (70, 200),
        "Shuffled 1 (200-399)": (200, 400),
        "Shuffled 2 (400-600)": (400, 601),
    }
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]

    for i, (phase_name, (start, end)) in enumerate(phases.items()):
        mask = [(s >= start and s < end) for s in steps]
        phase_steps = [s for s, m in zip(steps, mask) if m]
        phase_judge = [j for j, m in zip(judge, mask) if m]
        phase_tools = [t for t, m in zip(tools, mask) if m]
        phase_eff = [e for e, m in zip(eff, mask) if m]

        if not phase_steps:
            continue

        fig4.add_trace(go.Box(y=phase_judge, name=phase_name, marker_color=colors[i],
                              showlegend=False), row=1, col=1)
        fig4.add_trace(go.Box(y=phase_tools, name=phase_name, marker_color=colors[i],
                              showlegend=False), row=1, col=2)
        fig4.add_trace(go.Box(y=phase_eff, name=phase_name, marker_color=colors[i],
                              showlegend=False), row=2, col=1)

    # 4d: Clipped ratio (truncation)
    clipped = np.array([m["completions/clipped_ratio"] for m in metrics])
    fig4.add_trace(go.Scatter(x=steps, y=smooth(clipped, 10), mode="lines", name="Clipped ratio (MA-10)",
                              line=dict(color="red", width=2)), row=2, col=2)

    fig4.update_layout(height=700, width=1100, title_text="GRPO Training — Curriculum Phase Analysis",
                       showlegend=False, template="plotly_white")

    # === Save all figures ===
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1.write_html(str(out_dir / "01_rewards.html"))
    fig2.write_html(str(out_dir / "02_dynamics.html"))
    fig3.write_html(str(out_dir / "03_reward_hacking.html"))
    fig4.write_html(str(out_dir / "04_curriculum.html"))

    print(f"Saved 4 dashboards to {out_dir}/")

    # === Print summary stats ===
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY (600 steps)")
    print("=" * 60)

    for phase_name, (start, end) in phases.items():
        phase_j = [m["rewards/llm_judge_reward/mean"] for m in metrics if start <= m["step"] < end]
        phase_r = [m["reward"] for m in metrics if start <= m["step"] < end]
        phase_e = [m["rewards/efficiency_reward/mean"] for m in metrics if start <= m["step"] < end]
        phase_f = [m["rewards/format_reward/mean"] for m in metrics if start <= m["step"] < end]
        phase_t = [m["tools/call_frequency"] for m in metrics if start <= m["step"] < end]
        if not phase_j:
            continue
        print(f"\n{phase_name}:")
        print(f"  Judge:      {np.mean(phase_j):.3f} +/- {np.std(phase_j):.3f}  (min={np.min(phase_j):.3f}, max={np.max(phase_j):.3f})")
        print(f"  Reward:     {np.mean(phase_r):.3f} +/- {np.std(phase_r):.3f}")
        print(f"  Efficiency: {np.mean(phase_e):.3f} +/- {np.std(phase_e):.3f}")
        print(f"  Format:     {np.mean(phase_f):.3f} +/- {np.std(phase_f):.3f}")
        print(f"  Tool calls: {np.mean(phase_t):.2f} +/- {np.std(phase_t):.2f}")

    # Reward hacking indicators
    print("\n" + "-" * 60)
    print("REWARD HACKING INDICATORS")
    print("-" * 60)

    # Check: high format but low judge (gaming submit without quality)
    high_fmt_low_judge = sum(1 for f, j in zip(fmt, judge) if f > 0.8 and j < 0.2)
    print(f"  High format (>0.8) + low judge (<0.2): {high_fmt_low_judge}/{len(fmt)} ({100*high_fmt_low_judge/len(fmt):.1f}%)")

    # Check: judge improving but efficiency/format flat or declining
    late_judge_mean = np.mean(judge[450:600])
    early_judge_mean = np.mean(judge[:100])
    print(f"  Judge improvement: {early_judge_mean:.3f} -> {late_judge_mean:.3f} ({late_judge_mean - early_judge_mean:+.3f})")

    late_eff_mean = np.mean(eff[450:600])
    early_eff_mean = np.mean(eff[:100])
    print(f"  Efficiency change: {early_eff_mean:.3f} -> {late_eff_mean:.3f} ({late_eff_mean - early_eff_mean:+.3f})")

    # Check: completion length collapsing (model learning to be short for easy reward)
    late_len_mean = np.mean(mean_len[450:600])
    early_len_mean = np.mean(mean_len[:100])
    print(f"  Length change:     {early_len_mean:.0f} -> {late_len_mean:.0f} tokens")

    # Check: entropy collapse (model becoming deterministic)
    late_ent = np.mean(np.array([m["entropy"] for m in metrics])[450:600])
    early_ent = np.mean(np.array([m["entropy"] for m in metrics])[:100])
    print(f"  Entropy change:    {early_ent:.4f} -> {late_ent:.4f}")

    # Check: always submitting same number of snippets
    if tito_patterns:
        submit_sizes = []
        for p in tito_patterns:
            for s in p["submits"]:
                # Parse snippet count from submit args
                try:
                    ids = s.strip("[]'\" ").split(",")
                    submit_sizes.append(len(ids))
                except:
                    pass
        if submit_sizes:
            print(f"  Submit sizes:      mean={np.mean(submit_sizes):.1f}, std={np.std(submit_sizes):.1f}, "
                  f"range=[{min(submit_sizes)}, {max(submit_sizes)}]")
            # If std is very low, model might be gaming by always submitting same pattern
            if np.std(submit_sizes) < 0.5:
                print("  WARNING: Very low variance in submit sizes - possible pattern gaming")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    metrics_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/qwen3-14b-grpo-v2/trainer_state_600.json"
    log_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/qwen3-14b-grpo-v2/train_run2.log"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results/training_viz"

    print(f"Loading metrics from {metrics_path}...")
    metrics = load_metrics(metrics_path)
    print(f"Loaded {len(metrics)} steps")

    build_dashboard(metrics, log_path, output_dir)
