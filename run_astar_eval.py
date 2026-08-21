
"""
run_astar_eval.py
 
ПРОМЕЖУТОЧНЫЙ метод (не финальный) — включён для полноты картины
экспериментов, описанных в отчёте. Многошаговое A*-планирование чисто
поверх FB-оценки (без графа реальных переходов). Лучшая найденная
конфигурация: hard-выбор (argmax), depth=2, смешанный пул кандидатов.
 
В отчёте эта конфигурация используется как демонстрация того, что
многошаговое планирование само по себе помогает (depth=1 -> depth=2
даёт устойчивый прирост), но чисто FB-based версия всё ещё уступает
финальному graph+FB методу (см. run_graph_eval.py) из-за шума
дискретного argmax по зашумлённой FB-оценке отдельных кандидатов.
 
Запуск:  python3 run_astar_eval.py
Из ВНУТРИ папки switching-successor-measures-main.
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
from astar_planning import build_candidate_anchor_pool, choose_subgoal_hard
 
ENV_NAME = "ogbench-antmaze-medium-navigate-v0"
MAX_DEPTH = 2
NUM_SEEDS = 20
TASK_ID = 1
MAX_STEPS = 1000
 
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
 
print(f"Вычисляем z для задачи (task_id={TASK_ID})...")
env.reset(options=dict(task_id=TASK_ID))
zero_shot_dataset = relabel_dataset(ENV_NAME, env, zero_shot_dataset_dict, complex_task_name=None)
zero_shot_dataset = HGCDataset(Dataset.create(**zero_shot_dataset), config)
num_zero_shot_samples = config.get('num_zero_shot_samples', 100_000)
zero_shot_batch = zero_shot_dataset.sample(
    num_zero_shot_samples, idxs=np.arange(num_zero_shot_samples),
    relabeling=False, augmentation=False,
)
z_goal = np.asarray(agent.infer_latent(zero_shot_batch))
 
print("Строим пул индексов-якорей...")
pool_idxs, pool_xy = build_candidate_anchor_pool(train_dataset, pool_size=2000)
 
 
def run_episode(seed, task_id=TASK_ID, max_steps=MAX_STEPS, replan_every=15,
                 stuck_eps=0.5, stuck_check_every=15, stuck_patience=30):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed, options=dict(task_id=task_id))
    done = False
    steps = 0
    current_z_w = None
    last_check_xy = np.asarray(obs)[:2].copy()
    stuck_until = -1
    rng = jax.random.PRNGKey(seed)
    t_start = time.time()
 
    while not done and steps < max_steps:
        if steps % stuck_check_every == 0:
            moved = np.linalg.norm(np.asarray(obs)[:2] - last_check_xy)
            if steps > 0 and moved < stuck_eps:
                stuck_until = steps + stuck_patience
            last_check_xy = np.asarray(obs)[:2].copy()
 
        if steps % replan_every == 0 and steps >= stuck_until:
            subgoal = choose_subgoal_hard(
                agent, np.asarray(obs), z_goal, train_dataset, config, pool_idxs, pool_xy,
                num_candidates=6, max_depth=MAX_DEPTH, max_expansions=8,
            )
            if subgoal is None:
                current_z_w = jnp.asarray(z_goal)[None]
            else:
                z_w = agent.network.select('backward_repr')(subgoal[None])
                current_z_w = agent.normalize_z(z_w)
 
        if steps < stuck_until:
            rng, sub_rng = jax.random.split(rng)
            action = agent.sample_actions(obs[None], jnp.asarray(z_goal)[None], seed=sub_rng, temperature=0)[0]
        else:
            z_w = current_z_w if current_z_w is not None else jnp.asarray(z_goal)[None]
            low_dist = agent.network.select('actor')(obs[None], z_w, goal_encoded=True, temperature=0)
            action = jnp.clip(low_dist.mode()[0], -1, 1)
 
        obs, reward, terminated, truncated, info = env.step(np.asarray(action))
        done = terminated or truncated
        steps += 1
 
    elapsed = time.time() - t_start
    success = info.get("success", 0)
    print(f"  сид {seed}: success={success}, шагов={steps}, время={elapsed:.0f} сек")
    return success
 
 
print(f"Прогоняем hard, depth={MAX_DEPTH} на {NUM_SEEDS} сидах, task_id={TASK_ID}...")
results = [run_episode(seed) for seed in range(NUM_SEEDS)]
 
n = len(results)
p = np.mean(results)
se = np.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0.0
print(f"\nИТОГО: {results}")
print(f"success rate = {p:.3f} (n={n})")
print(f"95% доверительный интервал: [{max(0,p - 1.96*se):.3f}, {min(1,p + 1.96*se):.3f}]")