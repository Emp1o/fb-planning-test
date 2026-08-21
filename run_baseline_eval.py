
"""
run_baseline_eval.py
 
Оценивает готовый бейзлайн (fbpiswitch high_actor, single-intention,
жадный выбор на каждом шаге) на antmaze-medium-navigate-v0, по всем
задачам среды, NUM_SEEDS_PER_TASK сидов на задачу.
 
Запуск:  python3 run_baseline_eval.py
Обязательно из ВНУТРИ папки switching-successor-measures-main, рядом
с папками agents/, utils/ и checkpoints/params_1000000.pkl.
"""
import json
import pickle
import numpy as np
import jax
import flax
 
from agents.fbpiswitch import FBpiSwitchAgent, get_config as get_agent_default_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
 
ENV_NAME = "ogbench-antmaze-medium-navigate-v0"
NUM_SEEDS_PER_TASK = 20   # итоговый success rate публикуется по всем задачам сразу
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
 
num_tasks = len(env.unwrapped.task_infos)
print(f"Всего задач в среде: {num_tasks}")
 
 
def get_inferred_latent(task_id):
    """z для конкретной задачи — проекция истинного reward на пространство B."""
    env.reset(options=dict(task_id=task_id))
    zero_shot_dataset = relabel_dataset(ENV_NAME, env, zero_shot_dataset_dict, complex_task_name=None)
    zero_shot_dataset = HGCDataset(Dataset.create(**zero_shot_dataset), config)
    num_zero_shot_samples = config.get('num_zero_shot_samples', 100_000)
    zero_shot_batch = zero_shot_dataset.sample(
        num_zero_shot_samples, idxs=np.arange(num_zero_shot_samples),
        relabeling=False, augmentation=False,
    )
    return np.asarray(agent.infer_latent(zero_shot_batch))
 
 
def run_episode(seed, task_id, inferred_latent, max_steps=MAX_STEPS):
    """Один эпизод бейзлайна. env.reset(seed=...) — критично для честного
    сравнения с graph+FB на тех же самых стартовых условиях."""
    obs, info = env.reset(seed=seed, options=dict(task_id=task_id))
    done = False
    steps = 0
    rng = jax.random.PRNGKey(seed)
 
    while not done and steps < max_steps:
        rng, sub_rng = jax.random.split(rng)
        action = agent.sample_actions(obs, inferred_latent, seed=sub_rng, temperature=0)
        action = np.clip(np.asarray(action), -1, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
 
    return info.get("success", 0)
 
 
all_results = []
per_task_results = {}
 
for task_id in range(1, num_tasks + 1):
    print(f"\n=== task_id={task_id}: вычисляем z... ===")
    inferred_latent = get_inferred_latent(task_id)
 
    task_results = []
    for seed in range(NUM_SEEDS_PER_TASK):
        success = run_episode(seed, task_id, inferred_latent)
        task_results.append(success)
    per_task_results[task_id] = task_results
    all_results.extend(task_results)
    print(f"  task_id={task_id}: success rate = {np.mean(task_results):.3f} (n={len(task_results)})")
 
print("\n=== ИТОГО ПО ВСЕМ ЗАДАЧАМ (бейзлайн) ===")
for task_id, results in per_task_results.items():
    print(f"  task_id={task_id}: {np.mean(results):.3f}")
 
n = len(all_results)
p = np.mean(all_results)
se = np.sqrt(p * (1 - p) / n)
print(f"\nОбщий success rate = {p:.3f} (n={n})")
print(f"95% доверительный интервал: [{p - 1.96*se:.3f}, {p + 1.96*se:.3f}]")