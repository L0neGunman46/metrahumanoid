# evaluate_hrl.py
import gymnasium as gym
import numpy as np
from metra.config import config
from metra.agent import METRAAgent
from metra.hrl_controller import HierarchicalController
from metra.logger import METRALogger


def create_downstream_tasks():
    # unchanged from your version
    tasks = {
        "run_forward": {
            "description": "Run forward as fast as possible",
            "reward_fn": lambda obs, action, next_obs: (next_obs[0] - obs[0]) * 10,
            "task_dim": 1,
            "task_info": np.array([1.0]),
            "success_fn": lambda final_x, initial_x: final_x - initial_x > 50,
        },
        "run_backward": {
            "description": "Run backward as fast as possible",
            "reward_fn": lambda obs, action, next_obs: (obs[0] - next_obs[0]) * 10,
            "task_dim": 1,
            "task_info": np.array([-1.0]),
            "success_fn": lambda final_x, initial_x: initial_x - final_x > 50,
        },
        "reach_position": {
            "description": "Reach target x position",
            "reward_fn": lambda obs, action, next_obs, target: -abs(next_obs[0] - target),
            "task_dim": 1,
            "task_info": None,
            "success_fn": lambda final_x, target_x: abs(final_x - target_x) < 5.0,
        },
    }
    return tasks


def _discrete_skill_index(z):
    # Index of the "one-hot"; if continuous, return None
    if z.ndim == 1 and np.sum(np.abs(z - z.mean())) > 0 and np.isclose(z.sum(), 0.0, atol=1e-6):
        return int(np.argmax(z))
    return None


def evaluate_hrl_task_with_logging(
    agent,
    hrl_controller,
    env,
    task_config,
    logger,
    task_name,
    training_episode=0,
    num_episodes=10,
):
    episode_rewards = []
    episode_data = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=config["seed"] + episode + training_episode * 1000)
        initial_x = state[0]
        episode_reward = 0.0
        skill_step_counter = 0
        current_skill = None
        skill_indices = []  # for discrete entropy
        hrl_losses = []

        if "reach_position" in task_name and task_config["task_info"] is None:
            target_x = float(np.random.uniform(-50, 50))
            task_info = np.array([target_x], dtype=np.float32)
        else:
            task_info = task_config["task_info"]
            target_x = None

        for step in range(env.spec.max_episode_steps):
            if skill_step_counter == 0:
                current_skill = hrl_controller.select_skill(state, task_info)
                idx = _discrete_skill_index(current_skill)
                if idx is not None:
                    skill_indices.append(idx)

            action = agent.select_action(state, current_skill)
            next_state, _, terminated, truncated, _ = env.step(action)

            if "target" in task_config["reward_fn"].__code__.co_varnames:
                reward = float(task_config["reward_fn"](state, action, next_state, task_info[0]))
            else:
                reward = float(task_config["reward_fn"](state, action, next_state))

            episode_reward += reward

            skill_step_counter += 1
            if skill_step_counter >= config["skill_length"]:
                loss = hrl_controller.update(
                    state, task_info, current_skill, reward, next_state, terminated or truncated
                )
                hrl_losses.append(loss)
                skill_step_counter = 0

            state = next_state
            if terminated or truncated:
                break

        final_x = state[0]
        if target_x is not None:
            success = bool(task_config["success_fn"](final_x, target_x))
        else:
            success = bool(task_config["success_fn"](final_x, initial_x))

        # Skill entropy (discrete): H over skill indices
        if len(skill_indices) > 0:
            counts = np.bincount(skill_indices, minlength=config["skill_dim"]).astype(np.float64)
            p = counts / (counts.sum() + 1e-8)
            skill_entropy = float(-(p * np.log(p + 1e-12)).sum())
        else:
            skill_entropy = 0.0

        log_metrics = {
            "task_name": task_name,
            "training_episode": training_episode,
            "evaluation_episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": step + 1,
            "hrl_controller_loss": float(np.mean(hrl_losses)) if hrl_losses else 0.0,
            "skill_entropy": skill_entropy,
            "avg_skill_duration": config["skill_length"],
            "task_success": success,
            "final_position": float(final_x),
        }
        logger.log_hrl_episode(log_metrics)

        episode_rewards.append(episode_reward)
        episode_data.append(
            {"reward": episode_reward, "success": success, "final_x": final_x, "steps": step + 1}
        )

        if episode < 5 or episode % 10 == 0:
            print(
                f"    Ep {episode+1:3d}: R={episode_reward:7.2f} | "
                f"Steps={step+1:3d} | {'✓' if success else '✗'}"
            )

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    success_rate = float(np.mean([ep["success"] for ep in episode_data])) if episode_data else 0.0

    print(
        f"  Summary: Avg Reward={avg_reward:7.2f}±{std_reward:5.2f} | "
        f"Success Rate={success_rate:.2%}"
    )

    return avg_reward, std_reward, success_rate, episode_data


def main():
    experiment_name = f"hrl_{config['env_id']}_seed{config['seed']}"
    logger = METRALogger(experiment_name=experiment_name)

    print("=" * 60)
    print("METRA HRL DOWNSTREAM TASK EVALUATION")
    print(f"Experiment: {experiment_name}")
    print("=" * 60)

    env = gym.make(config["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = METRAAgent(state_dim, action_dim, config)

    pretraining_name = f"metra_{config['env_id']}_seed{config['seed']}"
    # match your train.py save names
    model_path = f"results/models/{pretraining_name}_final.pt"

    try:
        agent.load(model_path)
        print(f"✓ Loaded pre-trained METRA agent from {model_path}")
    except FileNotFoundError:
        print(f"✗ Pre-trained agent not found at {model_path}!")
        print("Run train.py first to create the pre-trained model.")
        return

    for p in agent.phi.parameters():
        p.requires_grad = False
    for p in agent.actor.parameters():
        p.requires_grad = False
    print("✓ Frozen pre-trained networks (φ and π)")

    tasks = create_downstream_tasks()
    all_results = {}

    for task_name, task_cfg in tasks.items():
        print("\n" + "=" * 60)
        print(f"TASK: {task_name.upper()}")
        print(f"Description: {task_cfg['description']}")
        print("=" * 60)

        hrl_controller = HierarchicalController(
            state_dim=state_dim,
            skill_dim=config["skill_dim"],
            task_dim=task_cfg["task_dim"],
            config=config,
        )

        print("\n--- Training HRL Controller (100 episodes) ---")
        training_rewards = []
        for training_ep in range(100):
            avg_reward, _, success_rate, _ = evaluate_hrl_task_with_logging(
                agent,
                hrl_controller,
                env,
                task_cfg,
                logger,
                task_name,
                training_episode=training_ep,
                num_episodes=1,
            )
            training_rewards.append(avg_reward)
            if training_ep % 20 == 0:
                print(f"  Training Ep {training_ep:3d}: Avg Reward = {avg_reward:7.2f}")

        print("\n--- Final Evaluation (20 episodes) ---")
        final_avg, final_std, final_sr, final_eps = evaluate_hrl_task_with_logging(
            agent, hrl_controller, env, task_cfg, logger, task_name, training_episode=100, num_episodes=20
        )

        all_results[task_name] = {
            "avg_reward": final_avg,
            "std_reward": final_std,
            "success_rate": final_sr,
            "training_curve": training_rewards,
        }

        print(f"\n✓ Task {task_name} completed!")
        print(f"  Final Performance: {final_avg:.2f} ± {final_std:.2f}")
        print(f"  Success Rate: {final_sr:.2%}")

    print("\n" + "=" * 60)
    print("FINAL HRL EVALUATION SUMMARY")
    print("=" * 60)
    for task_name, res in all_results.items():
        print(
            f"{task_name:15s}: {res['avg_reward']:7.2f}±{res['std_reward']:5.2f} | "
            f"Success: {res['success_rate']:6.2%}"
        )

    summary_file = f"results/logs/{experiment_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("METRA HRL Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Environment: {config['env_id']}\n")
        f.write(f"Seed: {config['seed']}\n")
        f.write(f"Skill Dim: {config['skill_dim']}\n\nResults:\n")
        for task_name, res in all_results.items():
            f.write(
                f"{task_name}: {res['avg_reward']:.2f}±{res['std_reward']:.2f} "
                f"(Success: {res['success_rate']:.2%})\n"
            )

    print(f"\n✓ Summary saved: {summary_file}")
    print(f"✓ Detailed logs saved to results/logs/{experiment_name}_hrl.csv")
    env.close()


if __name__ == "__main__":
    main()