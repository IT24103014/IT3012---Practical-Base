# agent.py
import random
from collections import deque


class SimpleReflexAgent:
    """Condition-action agent: no stored history, only the current percept matters."""

    def sense_and_act(self, percept: dict) -> str:
        # Strict IF-THEN rules, with no history stored in __init__.
        if percept.get('food_here') or percept.get('smells_food'):
            return 'Up'
        if percept.get('wall_ahead') or percept.get('hit_wall'):
            return 'Left'
        return 'Right'


class ModelBasedAgent:
    """A memory-based agent that avoids repeating failed moves for the same percept."""

    def __init__(self):
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        wall_detected = percept.get('wall_ahead') or percept.get('hit_wall')
        food_detected = percept.get('food_here') or percept.get('smells_food')

        if food_detected:
            action = 'Up'
        elif wall_detected:
            preferred_order = ['Left', 'Right', 'Down', 'Up']
            if self.last_action in preferred_order:
                preferred_order.remove(self.last_action)
            action = preferred_order[0]
        else:
            action = 'Right'

        self.last_action = action
        return action


class SearchAgent:
    """Problem-solving agent that plans a path using BFS."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        rows, cols = grid_size
        walls = {tuple(wall) for wall in walls}

        if start == goal:
            return []
        if goal in walls:
            return None

        queue = deque([start])
        parent = {start: (None, None)}
        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }

        while queue:
            current = queue.popleft()
            for action, (dx, dy) in directions.items():
                next_pos = (current[0] + dx, current[1] + dy)
                if not (0 <= next_pos[0] < cols and 0 <= next_pos[1] < rows):
                    continue
                if next_pos in walls or next_pos in parent:
                    continue

                parent[next_pos] = (current, action)
                if next_pos == goal:
                    path = []
                    node = next_pos
                    while parent[node][0] is not None:
                        prev, action_taken = parent[node]
                        path.append(action_taken)
                        node = prev
                    path.reverse()
                    return path
                queue.append(next_pos)

        return None


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        if percept.get('smells_food') or percept.get('food_here'):
            return 'Up'
        return random.choice(self.actions_pool)