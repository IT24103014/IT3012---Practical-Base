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
    """A memory-based agent that updates its internal state before acting."""

    def __init__(self):
        self.last_action = None
        self.last_position = None
        self.visited_cells = set()
        self.action_history = []

    def _direction_to_delta(self, action: str):
        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }
        return directions.get(action, (0, 0))

    def sense_and_act(self, percept: dict) -> str:
        # Transition and sensor model update: remember the current state and the previous move.
        pos = percept.get('agent_pos')
        if pos is not None:
            current_pos = tuple(pos)
            if self.last_position is not None:
                self.visited_cells.add(self.last_position)
            self.last_position = current_pos

        if self.last_action is not None:
            self.action_history.append(self.last_action)

        wall_detected = percept.get('wall_ahead') or percept.get('hit_wall')
        food_detected = percept.get('food_here') or percept.get('smells_food')

        if food_detected:
            action = 'Up'
        elif wall_detected:
            # Memory-guided rule: if a wall is ahead, avoid repeating the last move and
            # prefer directions that are not already in the recent memory.
            preferred_order = ['Left', 'Right', 'Down', 'Up']
            if self.last_action in preferred_order:
                preferred_order.remove(self.last_action)

            # Example memory-aware logic: if a cell has been visited, prefer another turn.
            for candidate in preferred_order:
                if self.last_action == candidate:
                    continue
                if candidate not in self.action_history[-4:]:
                    action = candidate
                    break
            else:
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
