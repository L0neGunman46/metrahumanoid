import torch

config = {
    # Environment and Seed
    "env_id": "Ant-v5",  # Changed to Ant-v5
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Training Parameters
    "num_train_steps": 1_000_000,    # Ant usually needs more steps than HalfCheetah
    "start_timesteps": 10_000,       # Warmup period
    "replay_buffer_size": 1_000_000, # Sufficient buffer for 1M steps
    "batch_size": 256,
    "trans_optimization_epochs": 50, # Number of gradient updates per environment step

    # Network Architecture
    "hidden_dim": 1024,              # Recommended for more complex environments
    "skill_dim": 2,                  # Skill dimension for Ant (continuous in paper)
    "discrete_skills": False,        # Changed to False for continuous skills

    # SAC Hyperparameters
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.01,

    # METRA-specific Hyperparameters
    "dual_reg": True,
    "dual_lam_init": 30.0,
    "dual_slack": 1e-3,
    "unit_length_skill": True,       # Normalize continuous skill vectors

    # HRL Parameters
    "skill_length": 50,              # How many steps a skill is executed before resampling

    # Logging and Evaluation
    "log_interval": 50_000,          # Log training curves every 50k steps
    "eval_interval": 100_000,        # Run evaluation and save models/plots every 100k steps
    "num_eval_skills": 8,            # Number of skills to evaluate/visualize

    # Goal reaching
    "enable_goal_reaching": True,
}