# metra/goal_reaching.py
import numpy as np
import torch
import gymnasium as gym # Imported for environment spec, not rendering

def _xy_from_env(env):
    if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "data"):
        qpos = env.unwrapped.sim.data.qpos
        if "HalfCheetah" in env.spec.id:
            return np.array([float(qpos[0])], dtype=np.float32)
        elif qpos.shape[0] >= 2:
            return np.array([float(qpos[0]), float(qpos[1])], dtype=np.float32)
    # fallback...
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
    if obs is not None:
        if "HalfCheetah" in env.spec.id:
            return np.array([float(obs[0])], dtype=np.float32)
        return np.array([float(obs[0]), float(obs[1])], dtype=np.float32)
    return np.zeros(1 if "HalfCheetah" in env.spec.id else 2, dtype=np.float32)

def evaluate_goal_reaching(agent, env, config, logger, training_step, num_goals=10):
    """Zero-shot goal reaching with CSV logging, adapted for HalfCheetah and Ant."""
    success_count = 0
    distances = []
    
    env_id = config["env_id"]
    
    print(f"  Starting Zero-Shot Goal Reaching for {env_id}...")

    for goal_id in range(num_goals):
        goal_x = None # Initialize for clarity
        goal_y = None
        target_goal_position = None # To store (x) or (x,y)
        current_agent_position = None # To store (x) or (x,y)
        state, _ = env.reset(seed=config["seed"] + goal_id)
        total_reward = 0
        skill = None # Skill will be updated periodically
        # 1. Sample Random Goal based on Environment
        if "HalfCheetah" in env_id:
            goal_x = float(np.random.uniform(-100, 100))
            target_goal_position = np.array([goal_x], dtype=np.float32)
            goal_state = np.array(state, copy=True)  # start from current obs
            goal_state[0] = goal_x
            goal_state = goal_state.astype(np.float32)
            
        elif "Ant" in env_id:
            goal_x = float(np.random.uniform(-50, 50))
            goal_y = float(np.random.uniform(-50, 50))
            target_goal_position = np.array([goal_x, goal_y], dtype=np.float32)
            goal_state = np.array(state, copy=True)
            goal_state[0], goal_state[1] = goal_x, goal_y
            goal_state = goal_state.astype(np.float32)
            
        else:
            # Fallback for other environments if needed, but for now focusing on cheetah/ant
            goal_state = env.observation_space.sample() # Sample a random full state
            target_goal_position = goal_state[:2] # Assuming (x,y) for generic plot, adjust as needed
            print(f"Warning: Goal sampling not specifically defined for {env_id}. Using generic approach.")
            
       

        for step in range(env.spec.max_episode_steps):
            # 2. Select Skill for Goal Reaching
            # Update skill every `skill_length` steps (or once at start if skill_length is large)
            if skill is None or step % config.get("skill_length", 10) == 0:
                skill = agent.select_skill_for_goal(state, goal_state)
            
            action = agent.select_action(state, skill)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # 3. Calculate Final Distance and Success
        final_pos = _xy_from_env(env)
        if "HalfCheetah" in env_id:
            current_agent_position = final_pos  # shape (1,)
            distance = float(np.linalg.norm(current_agent_position - target_goal_position))
            success = distance < 3.0
        elif "Ant" in env_id:
            current_agent_position = final_pos  # shape (2,)
            distance = float(np.linalg.norm(current_agent_position - target_goal_position))
            success = distance < 3.0
        else:
            current_agent_position = final_pos
            distance = float(np.linalg.norm(current_agent_position - target_goal_position))
            success = distance < 3.0


        distances.append(distance)
        if success:
            success_count += 1
        
        # Log to CSV
        log_metrics = {
            'training_step': training_step,
            'goal_id': goal_id,
            'goal_x': target_goal_position[0] if target_goal_position is not None else 0, # Log first dim of goal
            'final_x': current_agent_position[0] if current_agent_position is not None else 0, # Log first dim of final pos
            'distance_to_goal': distance,
            'success': success,
            'episode_steps': step + 1,
            'total_reward': total_reward
        }
        
        logger.log_zeroshot_episode(log_metrics)
        
        # Console output for this goal
        if "HalfCheetah" in env_id:
             print(f"  Goal {goal_id+1:2d}: Target x={target_goal_position[0]:6.1f} → Final x={current_agent_position[0]:6.1f} | "
                   f"Dist={distance:5.2f} | {'✓' if success else '✗'}")
        elif "Ant" in env_id:
             print(f"  Goal {goal_id+1:2d}: Target (x,y)=({target_goal_position[0]:.1f},{target_goal_position[1]:.1f}) → Final (x,y)=({current_agent_position[0]:.1f},{current_agent_position[1]:.1f}) | "
                   f"Dist={distance:5.2f} | {'✓' if success else '✗'}")
        else:
            print(f"  Goal {goal_id+1:2d}: Dist={distance:5.2f} | {'✓' if success else '✗'}")

    # Final summary for zero-shot
    success_rate = success_count / num_goals
    avg_distance = np.mean(distances)
    
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Distance: {avg_distance:.2f}")
    
    return success_rate