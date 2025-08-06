# metra/utils.py
import gymnasium as gym
import numpy as np
import torch
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_xy_from_env(env):
    # Works for MuJoCo-based Gymnasium envs: use sim.data.qpos[:2] when present
    qpos = None
    if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "data"):
        qpos = env.unwrapped.sim.data.qpos
    elif hasattr(env.unwrapped, "data") and hasattr(env.unwrapped.data, "qpos"):
        qpos = env.unwrapped.data.qpos
    if qpos is not None and qpos.shape[0] >= 2:
        return np.array([qpos[0], 0.0 if qpos.shape[0] < 2 else qpos[1]], dtype=np.float32)
    # Fallback: try state[0:2]
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
    if obs is not None and obs.shape[0] >= 2:
        return np.array([obs[0], obs[1]], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def evaluate_and_visualize(agent, config, training_step):
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/trajectories", exist_ok=True)

    eval_env = gym.make(config["env_id"])
    device = config["device"]
    num_skills = int(config.get("num_eval_skills", 8))
    seed = int(config["seed"])

    print(f"\n--- Evaluating {num_skills} skills at step {training_step} ---")
    colors = plt.cm.hsv(np.linspace(0, 1, num_skills))

    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
    ax_traj.set_title(f"Skill Trajectories (Step {training_step})")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.grid(True)

    fig_latent, ax_latent = plt.subplots(figsize=(8, 6))
    ax_latent.set_title(f"Latent Space φ(s) (Step {training_step})")
    ax_latent.set_xlabel("φ1")
    ax_latent.set_ylabel("φ2")
    ax_latent.grid(True)

    trajectories_data = {
        "training_step": training_step,
        "environment": config["env_id"],
        "skills": [],
    }

    final_x_positions = []

    for i in range(num_skills):
        z = agent.sample_skill()
        state, _ = eval_env.reset(seed=seed + i)

        xy = []
        lat = []

        for t in range(eval_env.spec.max_episode_steps):
            xy.append(_get_xy_from_env(eval_env))

            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                phi_s = agent.phi(st).cpu().numpy().squeeze(0)
                lat.append(phi_s)

            action = agent.select_action(state, z)
            state, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break

        xy = np.array(xy, dtype=np.float32)
        lat = np.array(lat, dtype=np.float32)

        if xy.shape[0] > 0:
            final_x_positions.append(float(xy[-1, 0]))

        ax_traj.plot(xy[:, 0], xy[:, 1], color=colors[i], linewidth=1.5, label=f"Skill {i+1}")
        ax_traj.scatter(xy[0, 0], xy[0, 1], color=colors[i], marker="o", s=24)
        ax_traj.scatter(xy[-1, 0], xy[-1, 1], color=colors[i], marker="x", s=36)

        if lat.shape[0] > 0 and lat.shape[1] >= 2:
            ax_latent.plot(lat[:, 0], lat[:, 1], color=colors[i], linewidth=1.2)
            ax_latent.scatter(lat[0, 0], lat[0, 1], color=colors[i], marker="o", s=24)
            ax_latent.scatter(lat[-1, 0], lat[-1, 1], color=colors[i], marker="x", s=36)

        trajectories_data["skills"].append(
            {
                "skill_id": i,
                "skill_vector": np.asarray(z, dtype=np.float32).tolist(),
                "xy_trajectory": xy.tolist(),
                "latent_trajectory": lat.tolist(),
                "final_x": float(xy[-1, 0]) if xy.shape[0] > 0 else 0.0,
                "episode_length": int(xy.shape[0]),
            }
        )

    diversity = float(np.std(final_x_positions)) if final_x_positions else 0.0
    print(f"X-position diversity: {diversity:.3f}")

    ax_traj.legend(loc="best", fontsize=8)
    traj_filename = f"results/plots/skill_trajectories_step_{training_step}.png"
    fig_traj.savefig(traj_filename, dpi=150, bbox_inches="tight")
    plt.close(fig_traj)

    if len(trajectories_data["skills"]) > 0 and len(trajectories_data["skills"][0]["latent_trajectory"]) > 0:
        latent_filename = f"results/plots/latent_space_step_{training_step}.png"
        fig_latent.savefig(latent_filename, dpi=150, bbox_inches="tight")
    plt.close(fig_latent)

    data_filename = f"results/trajectories/skills_step_{training_step}.json"
    with open(data_filename, "w") as f:
        json.dump(trajectories_data, f, indent=2)

    print(f"✓ Plots saved: {traj_filename}")
    print(f"✓ Data saved: {data_filename}")
    eval_env.close()
    return diversity