"""
astar_planning.py
 
Базовые функции для оценки состояний через FB-представления (F, B) и
для A*-поиска по сабгоалам поверх них. Эти функции переиспользуются
graph_navigator.py — там F/B используются для выбора лучшего кандидата
внутри ячейки графа, а не для поиска пути целиком.
 
ВАЖНОЕ ИСПРАВЛЕНИЕ (найдено в процессе экспериментов):
forward_repr в чекпоинте — ансамбль из num_ensembles=2 сетей (см.
agents/fbpiswitch.py, ensemblize(..., num_ensembles=2)). Сырой выход
agent.successor_measure_extract(...) поэтому имеет форму (2, n), а не
(n,). Наивный reshape(-1) "распрямляет" два независимых ансамблевых
предсказания в 2n значений, что даёт неверную арифметику при их
дальнейшем комбинировании. Здесь используется корректная редукция —
консервативный минимум по оси ансамбля (как в clipped double-Q).
"""
import heapq
import itertools
import numpy as np
import jax.numpy as jnp
 
 
class Node:
    """Узел дерева поиска для astar_search."""
    __slots__ = ["obs", "g", "depth", "parent", "root_child_id"]
 
    def __init__(self, obs, g, depth, parent, root_child_id):
        self.obs = obs
        self.g = g
        self.depth = depth
        self.parent = parent
        self.root_child_id = root_child_id
 
 
def build_candidate_anchor_pool(train_dataset, pool_size=2000):
    """
    Пул РЕАЛЬНЫХ индексов датасета + их x,y — используется, чтобы найти
    индекс, ближайший к текущей позиции агента, и от него получить
    реалистичные сабгоалы через train_dataset.sample_goals(...).
    """
    idxs = train_dataset.dataset.get_random_idxs(pool_size)
    obs = np.asarray(train_dataset.get_observations(idxs))
    xy = obs[:, :2]
    return idxs, xy
 
 
def nearest_anchor_idx(pool_idxs, pool_xy, query_obs):
    dists = np.linalg.norm(pool_xy - query_obs[:2][None, :], axis=1)
    return int(pool_idxs[np.argmin(dists)])
 
 
def realistic_candidates(train_dataset, config, anchor_idx, num_candidates):
    """
    Сабгоалы из ТОЙ ЖЕ смеси (curgoal/trajgoal/randomgoal), на которой
    обучался high_actor — не выдуманные точки, а реальные состояния из
    датасета, привязанные к anchor_idx.
    """
    anchor_idxs = np.full(num_candidates, anchor_idx)
    goal_idxs = train_dataset.sample_goals(
        anchor_idxs,
        config['value_p_curgoal'],
        config['value_p_trajgoal'],
        config['value_p_randomgoal'],
        config['value_geom_sample'],
    )
    return np.asarray(train_dataset.get_observations(goal_idxs))
 
 
def _reduce_ensemble(raw):
    """forward_repr — ансамбль из 2 сетей, сырой выход (2, n). Минимум по оси 0."""
    raw = np.asarray(raw)
    if raw.ndim == 1:
        return raw
    return raw.min(axis=0)
 
 
def heuristic_batch(agent, obs_batch, z_goal):
    """
    h(w) = V^pi(w; z_goal) — "если бы прямо отсюда пойти к цели z,
    минуя все промежуточные сабгоалы". Один вызов на весь батч кандидатов.
    """
    n = obs_batch.shape[0]
    z_goal_b = jnp.broadcast_to(jnp.asarray(z_goal), (n, z_goal.shape[-1]))
    val = agent.successor_measure_extract(jnp.asarray(obs_batch), z_goal_b, z_goal_b)
    return _reduce_ensemble(val)
 
 
def edge_advantage_batch(agent, obs_from, obs_to_batch, z_goal):
    """
    A(s, w, z) — switching advantage (Corollary 1 статьи): насколько
    выгоднее сначала зайти в w, чем идти напрямую к z. Батчированная
    версия — все кандидаты w оцениваются одним вызовом сети.
    """
    n = obs_to_batch.shape[0]
    obs_from_b = jnp.broadcast_to(jnp.asarray(obs_from), (n, obs_from.shape[-1]))
    z_goal_b = jnp.broadcast_to(jnp.asarray(z_goal), (n, z_goal.shape[-1]))
    obs_to_batch = jnp.asarray(obs_to_batch)
 
    latents = agent.network.select('backward_repr')(obs_to_batch)  # z_w = B(w)
 
    Msww   = _reduce_ensemble(agent.successor_measure_extract(obs_from_b, latents, latents))
    Mwww   = _reduce_ensemble(agent.successor_measure_extract(obs_to_batch, latents, latents))
    Vwrr   = _reduce_ensemble(agent.successor_measure_extract(obs_to_batch, z_goal_b, z_goal_b))
    Vrstar = _reduce_ensemble(agent.successor_measure_extract(obs_from_b, z_goal_b, z_goal_b))
    Vswr   = _reduce_ensemble(agent.successor_measure_extract(obs_from_b, z_goal_b, latents))
 
    adv = Vswr + Msww / Mwww * Vwrr - Vrstar
    return adv
 
 
def astar_search(agent, current_obs, z_goal, train_dataset, config, pool_idxs, pool_xy,
                  num_candidates=8, max_depth=2, max_expansions=10):
    """
    A*-поиск по сабгоалам с приоритетной очередью (f = g + h). Кандидаты
    на каждом уровне — realistic_candidates (смешанный пул). Возвращает
    для каждого варианта первого хода лучшую f, найденную в его поддереве.
    """
    counter = itertools.count()
    next_rc_id = itertools.count()
 
    root = Node(obs=current_obs, g=0.0, depth=0, parent=None, root_child_id=None)
    h_root = float(heuristic_batch(agent, current_obs[None], z_goal)[0])
    open_heap = [(-(root.g + h_root), next(counter), root)]
 
    root_child_obs = {}
    root_child_best_f = {}
    expansions = 0
 
    while open_heap and expansions < max_expansions:
        neg_f, _, node = heapq.heappop(open_heap)
        if node.depth >= max_depth:
            continue
        expansions += 1
        anchor_idx = nearest_anchor_idx(pool_idxs, pool_xy, node.obs)
        candidates = realistic_candidates(train_dataset, config, anchor_idx, num_candidates)
 
        step_advs = edge_advantage_batch(agent, node.obs, candidates, z_goal)
        h_news = heuristic_batch(agent, candidates, z_goal)
 
        for i in range(len(candidates)):
            g_new = node.g + float(step_advs[i])
            f_new = g_new + float(h_news[i])
            if node.depth == 0:
                rc_id = next(next_rc_id)
                root_child_obs[rc_id] = candidates[i]
            else:
                rc_id = node.root_child_id
            root_child_best_f[rc_id] = max(root_child_best_f.get(rc_id, -np.inf), f_new)
            child = Node(obs=candidates[i], g=g_new, depth=node.depth + 1,
                         parent=node, root_child_id=rc_id)
            heapq.heappush(open_heap, (-f_new, next(counter), child))
 
    if not root_child_best_f:
        return None, None
    return root_child_obs, root_child_best_f
 
 
def choose_subgoal_hard(agent, current_obs, z_goal, train_dataset, config, pool_idxs, pool_xy,
                         num_candidates=8, max_depth=2, max_expansions=10):
    """Возвращает OBS лучшего первого хода (argmax по f)."""
    root_child_obs, root_child_best_f = astar_search(
        agent, current_obs, z_goal, train_dataset, config, pool_idxs, pool_xy,
        num_candidates, max_depth, max_expansions,
    )
    if root_child_obs is None:
        return None
    best_id = max(root_child_best_f, key=root_child_best_f.get)
    return root_child_obs[best_id]