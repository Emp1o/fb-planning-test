
"""
run_graph_eval.py
 
ФИНАЛЬНЫЙ МЕТОД — оценивает graph+FB v5 (см. graph_navigator.py) на
antmaze-medium-navigate-v0, по всем задачам среды, NUM_SEEDS_PER_TASK
сидов на задачу.
 
Запуск:  python3 run_graph_eval.py
Из ВНУТРИ папки switching-successor-measures-main, рядом с astar_planning.py
и graph_navigator.py (тоже должны лежать в этой папке).
"""
import json
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax
 
from agents.fbpiswitch import FBpiSwitchAgent, get_config as get_agent_default_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from graph_navigator import build_transition_graph, compute_fixed_goal_cell, pick_subgoal
 
ENV_NAME = "ogbench-antmaze-medium-navigate-v0"
CELL_SIZE = 0.4
REPLAN_EVERY = 3
FINAL_APPROACH_RADIUS = 3.0
MAX_CANDIDATES_PER_CELL = 8
NUM_SEEDS_PER_TASK = 20
MAX_STEPS = 1000
STUCK_EPS = 0.3
STUCK_CHECK_EVERY = 20
STUCK_PATIENCE = 20
 
print("Загружаем конфиг, среду, датасет, агента...")
with open("checkpoints/flags.json") as f:
    saved_flags = json.load(f)
config = get_agent_default_config()
config.update(saved_flags["agent"])
 
env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(
    ENV_NAME, frame_stack=config['frame_stack'], add_info=True
)
train_dataset_raw = Dataset.create(**train_dataset_raw)
zero_shot_dataset_dict = (
    train_dataset_raw if val_dataset_raw is None else Dataset.create(**val_dataset_raw)
)
train_dataset = HGCDataset(train_dataset_raw, config)
example_batch = train_dataset.sample(1)
 
agent = FBpiSwitchAgent.create(seed=0, ex_batch=example_batch, config=config)
with open("checkpoints/params_1000000.pkl", "rb") as f:
    load_dict = pickle.load(f)
agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
 
num_tasks = len(env.unwrapped.task_infos)
print(f"Всего задач в среде: {num_tasks}")
 
print("Строим граф реальных переходов (один раз, общий для всех задач)...")
t0 = time.time()
graph, cell_to_obs = build_transition_graph(train_dataset, cell_size=CELL_SIZE, max_transitions=1_000_000)
print(f"  готово за {time.time()-t0:.0f} сек, узлов в графе: {len(graph)}")
 
 
def get_inferred_latent(task_id):
    env.reset(options=dict(task_id=task_id))
    zero_shot_dataset = relabel_dataset(ENV_NAME, env, zero_shot_dataset_dict, complex_task_name=None)
    zero_shot_dataset = HGCDataset(Dataset.create(**zero_shot_dataset), config)
    num_zero_shot_samples = config.get('num_zero_shot_samples', 100_000)
    zero_shot_batch = zero_shot_dataset.sample(
        num_zero_shot_samples, idxs=np.arange(num_zero_shot_samples),
        relabeling=False, augmentation=False,
    )
    return np.asarray(agent.infer_latent(zero_shot_batch))
 
 
def run_episode(seed, task_id, z_goal, max_steps=MAX_STEPS):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed, options=dict(task_id=task_id))
    true_goal_obs = np.asarray(info.get('goal'))
    goal_xy = true_goal_obs[:2]
    fixed_goal_cell = compute_fixed_goal_cell(graph, np.asarray(obs), goal_xy, CELL_SIZE)
 
    done = False
    steps = 0
    current_subgoal = None
    last_check_xy = np.asarray(obs)[:2].copy()
    stuck_until = -1
    rng = jax.random.PRNGKey(seed)
 
    while not done and steps < max_steps:
        if steps % STUCK_CHECK_EVERY == 0:
            moved = np.linalg.norm(np.asarray(obs)[:2] - last_check_xy)
            if steps > 0 and moved < STUCK_EPS:
                stuck_until = steps + STUCK_PATIENCE
            last_check_xy = np.asarray(obs)[:2].copy()
 
        if steps % REPLAN_EVERY == 0 and steps >= stuck_until:
            current_subgoal = pick_subgoal(
                agent, np.asarray(obs), fixed_goal_cell, true_goal_obs, goal_xy,
                graph, cell_to_obs, z_goal, CELL_SIZE,
                final_approach_radius=FINAL_APPROACH_RADIUS,
                max_candidates_per_cell=MAX_CANDIDATES_PER_CELL,
            )
 
        if steps < stuck_until or current_subgoal is None:
            # анти-стак фолбэк / нет валидного пути по графу -> бейзлайновый high_actor
            rng, sub_rng = jax.random.split(rng)
            action = agent.sample_actions(obs[None], jnp.asarray(z_goal)[None], seed=sub_rng, temperature=0)[0]
        else:
            z_w = agent.network.select('backward_repr')(current_subgoal[None])
            z_w = agent.normalize_z(z_w)
            low_dist = agent.network.select('actor')(obs[None], z_w, goal_encoded=True, temperature=0)
            action = jnp.clip(low_dist.mode()[0], -1, 1)
 
        obs, reward, terminated, truncated, info = env.step(np.asarray(action))
        done = terminated or truncated
        steps += 1
 
    return info.get("success", 0)
 
 
all_results = []
per_task_results = {}
 
for task_id in range(1, num_tasks + 1):
    print(f"\n=== task_id={task_id}: вычисляем z... ===")
    z_goal = get_inferred_latent(task_id)
 
    task_results = []
    for seed in range(NUM_SEEDS_PER_TASK):
        t0 = time.time()
        success = run_episode(seed, task_id, z_goal)
        task_results.append(success)
        print(f"    сид {seed}: success={success}, время={time.time()-t0:.0f} сек")
 
    per_task_results[task_id] = task_results
    all_results.extend(task_results)
    print(f"  task_id={task_id}: success rate = {np.mean(task_results):.3f} (n={len(task_results)})")
 
print("\n=== ИТОГО ПО ВСЕМ ЗАДАЧАМ (graph+FB v5) ===")
for task_id, results in per_task_results.items():
    print(f"  task_id={task_id}: {np.mean(results):.3f}")
 
n = len(all_results)
p = np.mean(all_results)
se = np.sqrt(p * (1 - p) / n)
print(f"\nОбщий success rate = {p:.3f} (n={n})")
print(f"95% доверительный интервал: [{p - 1.96*se:.3f}, {p + 1.96*se:.3f}]")