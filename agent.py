# agent.py
import random

class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos')
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """A simple reflex agent that reacts purely to immediate percepts using Condition-Action rules."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Up'  # Move to collect food or proceed
        elif percept.get('wall_ahead'):
            return 'Left'  # Turn/move left
        else:
            return 'Up'  # Move forward


class ModelBasedAgent:
    """A model-based agent that maintains an internal map and visited cells to escape loops."""

    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = 'Up'
        self.visited = {(0, 0)}
        self.walls_found = set()
        self.last_action = None
        self.last_was_wall_ahead = False

    def sense_and_act(self, percept: dict) -> str:
        # 1. Update Transition Model
        if self.last_action:
            prev_dir = self.direction
            self.direction = self.last_action
            
            # If we chose to move in the direction we were facing, and there was a wall,
            # we didn't move. Otherwise, we assume we successfully moved.
            if not (self.last_was_wall_ahead and self.last_action == prev_dir):
                if self.last_action == 'Up':
                    self.y += 1
                elif self.last_action == 'Down':
                    self.y -= 1
                elif self.last_action == 'Left':
                    self.x -= 1
                elif self.last_action == 'Right':
                    self.x += 1
                self.visited.add((self.x, self.y))

        # 2. Update Sensor Model (map walls)
        if percept.get('wall_ahead'):
            if self.direction == 'Up':
                self.walls_found.add((self.x, self.y + 1))
            elif self.direction == 'Down':
                self.walls_found.add((self.x, self.y - 1))
            elif self.direction == 'Left':
                self.walls_found.add((self.x - 1, self.y))
            elif self.direction == 'Right':
                self.walls_found.add((self.x + 1, self.y))

        # Save percept state for next transition update
        self.last_was_wall_ahead = percept.get('wall_ahead', False)

        # 3. Action Selection Rules
        # Order candidates: try to continue straight, then turn left/right, then reverse.
        opposites = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}
        opposite_dir = opposites.get(self.direction)

        if self.direction == 'Up':
            candidates = ['Up', 'Left', 'Right', 'Down']
        elif self.direction == 'Down':
            candidates = ['Down', 'Right', 'Left', 'Up']
        elif self.direction == 'Left':
            candidates = ['Left', 'Down', 'Up', 'Right']
        else:  # Right
            candidates = ['Right', 'Up', 'Down', 'Left']

        valid_candidates = []
        for action in candidates:
            # Calculate target coordinate
            if action == 'Up':
                target = (self.x, self.y + 1)
            elif action == 'Down':
                target = (self.x, self.y - 1)
            elif action == 'Left':
                target = (self.x - 1, self.y)
            else:  # Right
                target = (self.x + 1, self.y)

            if target not in self.walls_found:
                valid_candidates.append((action, target))

        if not valid_candidates:
            # Fallback
            action = 'Up'
        else:
            # Prioritize target cells that have not been visited
            unvisited_candidates = [c for c in valid_candidates if c[1] not in self.visited]
            if unvisited_candidates:
                action = unvisited_candidates[0][0]
            else:
                # If all are visited, choose the one that does not reverse direction if possible
                non_reverse = [c for c in valid_candidates if c[0] != opposite_dir]
                if non_reverse:
                    action = non_reverse[0][0]
                else:
                    action = valid_candidates[0][0]

        self.last_action = action
        return action


class SearchAgent:
    """A placeholder for the Search Agent to satisfy import and initial tests."""

    def bfs_search(self, start, goal, walls, grid_size):
        return []