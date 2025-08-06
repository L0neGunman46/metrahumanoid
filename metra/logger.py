# metra/logger.py
import csv
import os
import time
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class METRALogger:
    """Logging with headless plot generation + skill diversity plot."""
    def __init__(self, log_dir="results/logs", experiment_name=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        if experiment_name is None:
            experiment_name = f"metra_{int(time.time())}"
        self.experiment_name = experiment_name

        self.pretraining_file = os.path.join(log_dir, f"{experiment_name}_pretraining.csv")
        self.zeroshot_file = os.path.join(log_dir, f"{experiment_name}_zeroshot.csv")
        self.hrl_file = os.path.join(log_dir, f"{experiment_name}_hrl.csv")

        self._init_pretraining_csv()
        self._init_zeroshot_csv()
        self._init_hrl_csv()

        self.pretraining_metrics = defaultdict(list)
        self.zeroshot_metrics = defaultdict(list)

    def _init_pretraining_csv(self):
        headers = [
            "training_step",
            "episode_num",
            "episode_timesteps",
            "episode_reward",
            "intrinsic_reward",
            "phi_loss",
            "critic_loss",
            "actor_loss",
            "alpha_loss",
            "alpha",
            "dual_lambda",
            "constraint_violation",
            "q_value",
            "buffer_size",
            "wall_clock_time",
        ]
        with open(self.pretraining_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    def _init_zeroshot_csv(self):
        headers = [
            "training_step",
            "goal_id",
            "goal_x",
            "final_x",
            "distance_to_goal",
            "success",
            "episode_steps",
            "total_reward",
        ]
        with open(self.zeroshot_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    def _init_hrl_csv(self):
        headers = [
            "task_name",
            "training_episode",
            "evaluation_episode",
            "episode_reward",
            "episode_steps",
            "task_success",
            "final_position",
            "skill_entropy",
            "avg_skill_duration",
            "hrl_controller_loss",
        ]
        with open(self.hrl_file, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    def log_pretraining_step(self, metrics):
        for k, v in metrics.items():
            self.pretraining_metrics[k].append(v)
        row = [
            metrics.get("training_step", 0),
            metrics.get("episode_num", 0),
            metrics.get("episode_timesteps", 0),
            metrics.get("episode_reward", 0.0),
            metrics.get("intrinsic_reward", 0.0),
            metrics.get("phi_loss", 0.0),
            metrics.get("critic_loss", 0.0),
            metrics.get("actor_loss", 0.0),
            metrics.get("alpha_loss", 0.0),
            metrics.get("alpha", 0.0),
            metrics.get("dual_lambda", 0.0),
            metrics.get("constraint_violation", 0.0),
            metrics.get("q_value", 0.0),
            metrics.get("buffer_size", 0),
            time.time(),
        ]
        with open(self.pretraining_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def log_zeroshot_episode(self, metrics):
        for k, v in metrics.items():
            self.zeroshot_metrics[k].append(v)
        row = [
            metrics.get("training_step", 0),
            metrics.get("goal_id", 0),
            metrics.get("goal_x", 0.0),
            metrics.get("final_x", 0.0),
            metrics.get("distance_to_goal", 0.0),
            metrics.get("success", False),
            metrics.get("episode_steps", 0),
            metrics.get("total_reward", 0.0),
        ]
        with open(self.zeroshot_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def log_hrl_episode(self, metrics):
        row = [
            metrics.get("task_name", ""),
            metrics.get("training_episode", 0),
            metrics.get("evaluation_episode", 0),
            metrics.get("episode_reward", 0.0),
            metrics.get("episode_steps", 0),
            metrics.get("task_success", False),
            metrics.get("final_position", 0.0),
            metrics.get("skill_entropy", 0.0),
            metrics.get("avg_skill_duration", 0.0),
            metrics.get("hrl_controller_loss", 0.0),
        ]
        with open(self.hrl_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def plot_pretraining_curves(self):
        if not self.pretraining_metrics:
            return
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"METRA Pretraining - {self.experiment_name}")

        if "intrinsic_reward" in self.pretraining_metrics:
            axes[0, 0].plot(self.pretraining_metrics["intrinsic_reward"])
            axes[0, 0].set_title("Intrinsic Reward")
            axes[0, 0].grid(True)

        if "phi_loss" in self.pretraining_metrics:
            axes[0, 1].plot(self.pretraining_metrics["phi_loss"], label="phi")
            if "critic_loss" in self.pretraining_metrics:
                axes[0, 1].plot(self.pretraining_metrics["critic_loss"], label="critic")
            if "actor_loss" in self.pretraining_metrics:
                axes[0, 1].plot(self.pretraining_metrics["actor_loss"], label="actor")
            axes[0, 1].legend()
            axes[0, 1].set_title("Losses")
            axes[0, 1].grid(True)

        if "constraint_violation" in self.pretraining_metrics:
            axes[0, 2].plot(self.pretraining_metrics["constraint_violation"])
            axes[0, 2].set_title("Constraint Violation (avg)")
            axes[0, 2].grid(True)

        if "dual_lambda" in self.pretraining_metrics:
            axes[1, 0].plot(self.pretraining_metrics["dual_lambda"])
            axes[1, 0].set_title("Dual λ")
            axes[1, 0].grid(True)

        if "alpha" in self.pretraining_metrics:
            axes[1, 1].plot(self.pretraining_metrics["alpha"])
            axes[1, 1].set_title("SAC α")
            axes[1, 1].grid(True)

        if "episode_reward" in self.pretraining_metrics:
            axes[1, 2].plot(self.pretraining_metrics["episode_reward"])
            axes[1, 2].set_title("Episode Reward")
            axes[1, 2].grid(True)

        plt.tight_layout()
        out = f"results/plots/{self.experiment_name}_pretraining_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Training curves saved: {out}")

    def plot_zeroshot_summary(self):
        if not self.zeroshot_metrics or "success" not in self.zeroshot_metrics:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        steps = np.array(self.zeroshot_metrics["training_step"])
        succ = np.array(self.zeroshot_metrics["success"]).astype(np.float32)
        uniq = sorted(set(steps.tolist()))
        success_rates = [float(np.mean(succ[steps == s])) for s in uniq]
        ax1.plot(uniq, success_rates, marker="o")
        ax1.set_title("Zero-Shot Success Rate")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1)
        ax1.grid(True)

        distances = np.array(self.zeroshot_metrics["distance_to_goal"], dtype=np.float32)
        ax2.hist(distances, bins=20, edgecolor="black")
        ax2.set_title("Distance to Goal")
        ax2.grid(True)

        plt.tight_layout()
        out = f"results/plots/{self.experiment_name}_zeroshot_summary.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Zero-shot summary saved: {out}")

    def print_summary(self, phase="pretraining"):
        if phase == "pretraining" and self.pretraining_metrics:
            print("\n" + "=" * 50)
            print("PRETRAINING SUMMARY")
            print("=" * 50)
            if "intrinsic_reward" in self.pretraining_metrics:
                recent = np.mean(self.pretraining_metrics["intrinsic_reward"][-100:])
                print(f"Recent Intrinsic Reward: {recent:.4f}")
            if "constraint_violation" in self.pretraining_metrics:
                v = np.mean(self.pretraining_metrics["constraint_violation"][-100:])
                print(f"Recent Constraint Violation: {v:.4f}")
            self.plot_pretraining_curves()
        elif phase == "zeroshot":
            self.plot_zeroshot_summary()