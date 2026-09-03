# agent.py
import random
from collections import deque
import heapq

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
    """A search agent that can use BFS, DFS, and UCS to find paths on a grid."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'
        self.x = 0
        self.y = 0

    def sense_and_act(self, percept: dict) -> str:
        if 'agent_pos' in percept:
            self.x, self.y = percept['agent_pos']

        if not self.plan:
            current_pos = (self.x, self.y)
            food_list = percept.get('all_food', [])
            
            if not food_list:
                return 'Up'
                
            closest_food = min(
                food_list,
                key=lambda f: abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
            )
            
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (10, 10))
            
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(current_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(current_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(current_pos, closest_food, walls, grid_size)
            else:
                self.plan = []
                
        if not self.plan:
            return 'Up'
            
        action = self.plan.pop(0)
        
        if action == 'Up':
            self.y += 1
        elif action == 'Down':
            self.y -= 1
        elif action == 'Left':
            self.x -= 1
        elif action == 'Right':
            self.x += 1
            
        return action

    def bfs_search(self, start, goal, walls, grid_size):
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return []
        
        walls_set = {tuple(w) for w in walls}
        width, height = grid_size
        
        frontier = deque([(start, [])])
        reached = {start}
        
        actions = ['Up', 'Down', 'Left', 'Right']
        action_offsets = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        
        while frontier:
            state, path = frontier.popleft()
            
            if state == goal:
                return path
                
            for action in actions:
                dx, dy = action_offsets[action]
                next_state = (state[0] + dx, state[1] + dy)
                
                if 0 <= next_state[0] < width and 0 <= next_state[1] < height:
                    if next_state not in walls_set and next_state not in reached:
                        if next_state == goal:
                            return path + [action]
                        reached.add(next_state)
                        frontier.append((next_state, path + [action]))
                        
        return []

    def dfs_search(self, start, goal, walls, grid_size):
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return []
            
        walls_set = {tuple(w) for w in walls}
        width, height = grid_size
        
        frontier = [(start, [])]
        reached = set()
        
        actions = ['Up', 'Down', 'Left', 'Right']
        action_offsets = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        
        while frontier:
            state, path = frontier.pop()
            
            if state == goal:
                return path
                
            if state not in reached:
                reached.add(state)
                for action in actions:
                    dx, dy = action_offsets[action]
                    next_state = (state[0] + dx, state[1] + dy)
                    
                    if 0 <= next_state[0] < width and 0 <= next_state[1] < height:
                        if next_state not in walls_set and next_state not in reached:
                            frontier.append((next_state, path + [action]))
                            
        return []

    def ucs_search(self, start, goal, walls, grid_size):
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return []
            
        walls_set = {tuple(w) for w in walls}
        width, height = grid_size
        
        pq = []
        heapq.heappush(pq, (0, start, []))
        reached = {start: 0}
        
        actions = ['Up', 'Down', 'Left', 'Right']
        action_offsets = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        
        while pq:
            cost, state, path = heapq.heappop(pq)
            
            if state == goal:
                return path
                
            if cost > reached.get(state, float('inf')):
                continue
                
            for action in actions:
                dx, dy = action_offsets[action]
                next_state = (state[0] + dx, state[1] + dy)
                
                if 0 <= next_state[0] < width and 0 <= next_state[1] < height:
                    if next_state not in walls_set:
                        next_cost = cost + 1
                        if next_cost < reached.get(next_state, float('inf')):
                            reached[next_state] = next_cost
                            heapq.heappush(pq, (next_cost, next_state, path + [action]))
                            
        return []