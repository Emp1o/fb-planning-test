"""
graph_navigator.py
 
ФИНАЛЬНЫЙ МЕТОД (graph + FB, версия v5).
 
Идея: вместо того чтобы доверять argmax по (шумной) точечной FB-оценке
как единственному механизму "куда идти" (как в чисто FB-based A*),
строим ЧЕСТНЫЙ граф связности прямо из реальных переходов офлайн-
датасета: узлы — ячейки сетки по x,y, рёбра — только те пары ячеек,
между которыми зафиксирован хотя бы один реальный переход в датасете
(не пересекая границы эпизодов). Кратчайший путь по этому графу (BFS)
даёт физически осмысленный, огибающий стены маршрут — без привилегиро-
ванной карты стен симулятора, только из офлайн-данных.
 
Внутри одной целевой ячейки маршрута, если там несколько реальных
кандидатов, выбор лучшего среди них всё ещё делается через FB
(edge_advantage_batch/heuristic_batch из astar_planning.py) — то есть
граф отвечает за физическую достижимость, а FB — за то, какой из
достижимых вариантов ценнее для конкретной задачи.
 
История находок, приведших к этой версии (см. отчёт для деталей):
  v1: цель графа искалась заново на каждом шаге -> дрейф
  v2: критерий "дошёл до вейпоинта" не подходит FB-интенциям (агент
      проезжает мимо, а не паркуется) -> заменено на частый таймер
  v3: без анти-стака при истинном застревании эпизод "залипает"
  v4: цель фиксируется один раз в начале эпизода (bfs_all_reachable +
      nearest_reachable_cell) -> устраняет дрейф цели, но остаются
      колебания на подходе к цели
  v5: при приближении к истинной цели (< FINAL_APPROACH_RADIUS) граф
      больше не используется — π_l целится напрямую в истинную точку
"""
import numpy as np
from collections import defaultdict, deque
 
from astar_planning import edge_advantage_batch, heuristic_batch
 
 
def build_transition_graph(train_dataset, cell_size=0.4, max_transitions=1_000_000):
    """
    Строит граф РЕАЛЬНЫХ переходов из офлайн-датасета.
    cell_size=0.4 подобран эмпирически (крупнее — теряется разрешение
    коридоров лабиринта; крупный размер не улучшал результат в тестах).
    """
    ds = train_dataset.dataset
    n = min(len(ds['observations']), max_transitions)
 
    obs = np.asarray(ds['observations'][:n])
    terminals = np.asarray(ds['terminals'][:n]) if 'terminals' in ds else None
 
    xy = obs[:, :2]
    cells = np.floor(xy / cell_size).astype(int)
 
    graph = defaultdict(set)
    cell_to_obs = defaultdict(list)
 
    for i in range(n - 1):
        if terminals is not None and terminals[i]:
            continue  # не соединяем переходы через границу эпизода
        c1, c2 = tuple(cells[i]), tuple(cells[i + 1])
        cell_to_obs[c1].append(obs[i])
        if c1 != c2:
            graph[c1].add(c2)
            graph[c2].add(c1)
 
    return graph, cell_to_obs
 
 
def bfs_all_reachable(graph, start_cell):
    """Все клетки, реально достижимые из start_cell (обход в ширину)."""
    if start_cell not in graph:
        return {start_cell: 0}
    visited = {start_cell: 0}
    queue = deque([start_cell])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                queue.append(neighbor)
    return visited
 
 
def nearest_reachable_cell(graph, reachable_from_start, xy, cell_size):
    """
    Ближайшая по прямой клетка СРЕДИ РЕАЛЬНО ДОСТИЖИМЫХ из текущей
    позиции — а не просто ближайшая по прямой линии среди всех клеток
    графа (та версия давала дрейф цели на другую сторону стены).
    """
    target = np.floor(xy / cell_size).astype(int)
    if tuple(target) in reachable_from_start:
        return tuple(target)
    cells = np.array(list(reachable_from_start.keys()))
    if len(cells) == 0:
        return None
    dists = np.sum((cells - target[None, :]) ** 2, axis=1)
    return tuple(cells[np.argmin(dists)])
 
 
def shortest_path_next_cell(graph, start_cell, goal_cell):
    """BFS от start_cell до goal_cell. Возвращает первый хоп пути."""
    if start_cell not in graph or goal_cell not in graph:
        return None
    if start_cell == goal_cell:
        return start_cell
 
    visited = {start_cell}
    queue = [(start_cell, None)]
    idx = 0
    while idx < len(queue):
        current, first_hop = queue[idx]
        idx += 1
        for neighbor in graph[current]:
            nh = first_hop if first_hop is not None else neighbor
            if neighbor == goal_cell:
                return nh
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, nh))
    return None
 
 
def _nearest_cell_in_graph(graph, target):
    """Fallback: ближайшая по прямой клетка графа (для редкого случая,
    когда самая первая позиция агента ещё не покрыта графом)."""
    cells = np.array(list(graph.keys()))
    dists = np.sum((cells - np.array(target)[None, :]) ** 2, axis=1)
    return tuple(cells[np.argmin(dists)])
 
 
def compute_fixed_goal_cell(graph, start_obs, goal_xy, cell_size):
    """
    Вычисляется ОДИН РАЗ в начале эпизода (не на каждом шаге!) — иначе
    "реально достижимая замена истинной цели" может дрейфовать между
    двумя разными клетками от вызова к вызову (находка v3->v4).
    """
    start_target = tuple(np.floor(start_obs[:2] / cell_size).astype(int))
    start_cell = start_target if start_target in graph else _nearest_cell_in_graph(graph, start_target)
    reachable = bfs_all_reachable(graph, start_cell)
    return nearest_reachable_cell(graph, reachable, goal_xy, cell_size)
 
 
def pick_subgoal(agent, obs, fixed_goal_cell, true_goal_obs, goal_xy, graph, cell_to_obs,
                  z_goal, cell_size, final_approach_radius=3.0, max_candidates_per_cell=8):
    """
    Главная функция выбора сабгоала. Три случая:
      1. Агент близко к истинной цели -> целимся прямо в неё (устраняет
         колебания графового подхода на последних метрах, находка v4->v5).
      2. Есть путь по графу -> берём первый хоп, выбираем лучший из
         реальных кандидатов внутри той ячейки через FB.
      3. Пути нет / текущая позиция не в графе -> None (вызывающий код
         должен откатиться на бейзлайновый high_actor).
    """
    dist_to_goal = np.linalg.norm(obs[:2] - goal_xy)
    if dist_to_goal < final_approach_radius:
        return true_goal_obs
 
    cur_target = tuple(np.floor(obs[:2] / cell_size).astype(int))
    start_cell = cur_target if cur_target in graph else None
    if start_cell is None:
        return None
    next_cell = shortest_path_next_cell(graph, start_cell, fixed_goal_cell)
    if next_cell is None or next_cell not in cell_to_obs:
        return None
 
    all_candidates = cell_to_obs[next_cell]
    if len(all_candidates) > max_candidates_per_cell:
        # Ограничиваем число кандидатов, оцениваемых через FB на шаг —
        # иначе клетки с сотнями посещений делают каждый шаг средой
        # дорогим. Случайная подвыборка вместо использования всех точек.
        idx = np.random.choice(len(all_candidates), max_candidates_per_cell, replace=False)
        candidates = np.array(all_candidates)[idx]
    else:
        candidates = np.array(all_candidates)
 
    if len(candidates) == 1:
        return candidates[0]
 
    adv = edge_advantage_batch(agent, obs, candidates, z_goal)
    h = heuristic_batch(agent, candidates, z_goal)
    return candidates[int(np.argmax(adv + h))]