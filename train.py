# train.py
import gymnasium as gym
import numpy as np
import torch
import random
import time
import os
import matplotlib
matplotlib.use("Agg")

from metra.config import config
from metra.agent import METRAAgent
from metra.buffer import ReplayBuffer
from metra.utils import evaluate_and_visualize
from metra.goal_reaching import evaluate_goal_reaching
from metra.logger import METRALogger


def main():
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/trajectories", exist_ok=True)

    experiment_name = f"metra_{config['env_id']}_seed{config['seed']}"
    logger = METRALogger(experiment_name=experiment_name)

    print("METRA Training Started")
    print(f"Environment: {config['env_id']}")
    print(f"Device: {config['device']}")
    print("Plots will be saved to: results/plots/")
    print("-" * 50)

    env = gym.make(config["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = METRAAgent(state_dim, action_dim, config)
    replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    state, _ = env.reset(seed=config["seed"])
    skill = agent.sample_skill()
    skill_step_counter = 0

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    start_time = time.time()

    for t in range(1, config["num_train_steps"] + 1):
        episode_timesteps += 1
        skill_step_counter += 1

        if skill_step_counter >= config["skill_length"]:
            skill = agent.sample_skill()
            skill_step_counter = 0

        if t < config["start_timesteps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, skill)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.add(state, action, reward, next_state, done, skill)
        state = next_state
        episode_reward += float(reward)

        if done:
            update_logs = {}
            if len(replay_buffer) > config["batch_size"]:
                update_logs = agent.update(replay_buffer)

            log_metrics = {
                "training_step": t,
                "episode_num": episode_num + 1,
                "episode_timesteps": episode_timesteps,
                "episode_reward": episode_reward,
                "buffer_size": len(replay_buffer),
                **update_logs,
            }
            logger.log_pretraining_step(log_metrics)

            if episode_num % 10 == 0:
                ir = update_logs.get("intrinsic_reward", 0.0)
                print(f"Step {t:7d} | Ep {episode_num+1:4d} | R {episode_reward:8.2f} | IR {ir:6.3f}")

            state, _ = env.reset()
            skill = agent.sample_skill()
            skill_step_counter = 0
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        if t % config["log_interval"] == 0:
            elapsed = time.time() - start_time
            pct = 100.0 * t / config["num_train_steps"]
            print(f"\nProgress: {t}/{config['num_train_steps']} ({pct:.1f}%) | Time: {elapsed:.1f}s")
            logger.plot_pretraining_curves()
            start_time = time.time()

        if t % config["eval_interval"] == 0 and t > 0:
            print("\n" + "=" * 50)
            print(f"EVALUATION AT STEP {t}")
            print("=" * 50)

            evaluate_and_visualize(agent, config, training_step=t)

            # save checkpoint
            agent.save(f"results/models/metra_checkpoint_step_{t}.pt")

            if config.get("enable_goal_reaching", False):
                print("\n--- Zero-Shot Goal Reaching ---")
                evaluate_goal_reaching(agent, env, config, logger, t)
                logger.plot_zeroshot_summary()
            print("=" * 50 + "\n")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

    final_model_path = f"results/models/{experiment_name}_final.pt"
    agent.save(final_model_path)

    evaluate_and_visualize(agent, config, training_step=config["num_train_steps"])
    logger.print_summary("pretraining")
    logger.print_summary("zeroshot")

    print(f"✓ Final model: {final_model_path}")
    print("✓ Logs: results/logs/")
    print("✓ Plots: results/plots/")
    print("✓ Trajectories: results/trajectories/")

    env.close()


if __name__ == "__main__":
    main()