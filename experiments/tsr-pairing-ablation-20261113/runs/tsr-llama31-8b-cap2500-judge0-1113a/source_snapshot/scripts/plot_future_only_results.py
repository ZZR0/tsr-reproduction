#!/usr/bin/env python3
"""Plot the archived final evaluation and native training metrics. Requires matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    args = ap.parse_args()
    root = args.archive
    out = root / "figures"
    out.mkdir(exist_ok=True)
    results = json.loads((root / "metrics/results.json").read_text())
    by_name = {r["model"]:r for r in results["evaluations"]}
    names = ["M0", "Mf0", "M1", "Mf1", "M2_future_only"]
    labels = ["M0\nSFT", "Mf0", "M1", "Mf1", "M2\nfuture-only"]
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(names))
    for offset, key, color, title in [(-.19, "lc", "#2468a2", "Length-controlled win rate"),
                                     (.19, "soft_win", "#d47d24", "Soft win rate")]:
        bars = ax.bar(x+offset, [by_name[n][key] for n in names], .36, label=title, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Win rate (%)", ylim=(0, 36),
           title="AlpacaEval 2.0: 805 questions, generation seed 0")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "evaluation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title, ylabel in [(axes[0], "average_chars", "Answer length", "Mean characters"),
                                  (axes[1], "truncation_rate", "Output-limit truncation", "Answers (%)")]:
        values = [by_name[n][key] * (100 if key == "truncation_rate" else 1) for n in names]
        bars = ax.bar(x, values, color=["#8ba6bb"]*4+["#ce744f"])
        ax.bar_label(bars, fmt="%.2f" if key == "truncation_rate" else "%.0f", padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel)
        ax.margins(y=.2)
    save(fig, "length_and_truncation")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name in names[1:]:
        history = json.loads((root / f"training/{name}/trainer_state.json").read_text())["log_history"]
        for ax, key, title in zip(axes, ["loss", "rewards/margins", "rewards/accuracies"],
                                  ["DPO training loss", "Implicit reward margin", "Preference accuracy"]):
            points = [(r["step"], r[key]) for r in history if key in r]
            ax.plot(*zip(*points), label=name.replace("_future_only", " future-only"), linewidth=1.7)
            ax.set(xlabel="Optimizer updates within stage", title=title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Native training metrics; stages use different preference datasets", fontsize=11)
    save(fig, "dpo_training")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    hist = json.loads((root / "training/M0/trainer_state.json").read_text())["log_history"]
    for key, label in [("loss", "Train loss"), ("eval_loss", "SFT validation loss")]:
        points = [(r["step"], r[key]) for r in hist if key in r]
        ax.plot(*zip(*points), label=label, marker="o" if key == "eval_loss" else None)
    ax.set(xlabel="Optimizer updates", ylabel="Loss", title="SFT M0: 180 updates on 2 H200")
    ax.legend(frameon=False)
    save(fig, "sft_training")
    print(out)


if __name__ == "__main__":
    main()
