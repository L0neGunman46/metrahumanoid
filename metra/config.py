import torch

config = {
    "env_id": "Humanoid-v5",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_train_steps": 200_000,
    "start_timesteps": 10_000,
    "replay_buffer_size": 500_000,
    "batch_size": 256,
    "trans_optimization_epochs": 32,
    "hidden_dim": 512,        # temporarily smaller for speed
    "skill_dim": 2,
    "discrete_skills": False,
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.01,
    "dual_reg": True,
    "dual_lam_init": 30.0,
    "dual_slack": 1e-3,
    "unit_length_skill": True,
    "skill_length": 25,
    "log_interval": 25_000,
    "eval_interval": 100_000,
    "num_eval_skills": 8,
    "enable_goal_reaching": True,
}