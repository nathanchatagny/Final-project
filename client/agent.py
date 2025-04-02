import heapq
import math
import random
from base_agent import BaseAgent
from network import NetworkManager


BASE_DIRECTIONS = [
    (0, -1),  # Up
    (1, 0),  # Right
    (0, 1),  # Down
    (-1, 0),  # Left
]


class Agent(BaseAgent):
    def __init__(
        self,
        agent_name: str,
        network: NetworkManager,
        logger: str = "client.agent",
        is_dead: bool = True,
    ):
        """
        Initialize the agent
        Args:
            agent_name (str): The name of the agent
            network (NetworkManager): The network object to handle communication
            logger (str): The logger name
            is_dead (bool): Whether the agent is dead
        """
        # Initialize the base agent with the same parameters
        super().__init__(agent_name, network, logger, is_dead) 

        # You can access the base agent attributes and methods with "self" anywhere in the class. 
        # These attributes are automatically synchronized from the server data.
        # For example, here we log the agent name: 
        self.logger.info(f"Agent {self.agent_name} initialized")
        
        # Current target - will be either a passenger or the delivery zone
        self.current_target = None
        self.path = []
        
        # State management
        self.state = "COLLECT"  # "COLLECT" or "DELIVER"
        
        # Keep track of last position to detect when we've moved
        self.last_position = None
        
        # Emergency counter to unstuck the train
        self.stuck_counter = 0
        self.emergency_direction = None

    def get_direction(self):
        """
        This method is regularly called by the client to get the next direction of the train.
        Returns the optimal direction to move based on the current state and targets.
        """
        try:
            # Make sure we have all required data
            if (not self.all_trains or not self.agent_name in self.all_trains or 
                self.cell_size is None or self.delivery_zone is None):
                return BASE_DIRECTIONS[0]  # Default direction if data not available
            
            my_train = self.all_trains[self.agent_name]
            current_position = my_train.get("position")
            
            # Check if we're stuck
            if current_position == self.last_position:
                self.stuck_counter += 1
                if self.stuck_counter > 3:  # If stuck for too many cycles
                    # Try a random direction as emergency
                    if not self.emergency_direction or self.stuck_counter > 10:
                        self.emergency_direction = self._get_safe_random_direction(current_position)
                        self.stuck_counter = 4  # Reset but keep in "emergency mode"
                    return self.emergency_direction
            else:
                # We moved, reset stuck counter
                self.stuck_counter = 0
                self.emergency_direction = None
                
            # Update last position
            self.last_position = current_position
            
            # Check if we have many wagons and we're in deliver state - drop a wagon to boost speed
            if self.state == "DELIVER" and len(my_train.get("wagons", [])) > 5:
                self.network.send_drop_wagon_request()
            
            # Determine state
            if self.state == "COLLECT" and len(my_train.get("wagons", [])) >= 3:
                self.state = "DELIVER"
                self.path = []  # Clear path for new target
                self.current_target = None
            elif self.state == "DELIVER" and len(my_train.get("wagons", [])) == 0:
                self.state = "COLLECT"
                self.path = []  # Clear path for new target
                self.current_target = None
                
            # If we're in the delivery zone and have wagons, we're already at our target
            if self.state == "DELIVER" and self._is_in_delivery_zone(current_position):
                return my_train.get("direction")  # Keep current direction to stay in the zone
            
            # If we don't have a target or path, find one
            if not self.path:
                if self.state == "COLLECT":
                    # Find nearest passenger
                    self.current_target = self._find_nearest_passenger(current_position)
                else:  # DELIVER
                    # Target delivery zone center
                    delivery_pos = self.delivery_zone.get("position", (0, 0))
                    delivery_width = self.delivery_zone.get("width", 40)
                    delivery_height = self.delivery_zone.get("height", 40)
                    
                    # Target the center of the delivery zone
                    center_x = delivery_pos[0] + delivery_width // 2
                    center_y = delivery_pos[1] + delivery_height // 2
                    
                    # Adjust to grid alignment
                    center_x = (center_x // self.cell_size) * self.cell_size
                    center_y = (center_y // self.cell_size) * self.cell_size
                    
                    self.current_target = (center_x, center_y)
                
                if self.current_target:
                    self.path = self._find_path(current_position, self.current_target)
            
            # If we have a path, follow it
            if self.path:
                next_pos = self.path[0]
                
                # Calculate needed direction to get to next position
                direction = self._get_direction_to(current_position, next_pos)
                
                # If we're about to move, remove this step from the path
                if direction:
                    self.path.pop(0)
                    return direction
            
            # If we can't find a path, move randomly but safely
            return self._get_safe_random_direction(current_position)
            
        except Exception as e:
            self.logger.error(f"Error in get_direction: {e}")
            return BASE_DIRECTIONS[0]  # Default fallback
    
    def _find_nearest_passenger(self, position):
        """Find the nearest passenger to the given position"""
        if not self.passengers:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for passenger in self.passengers:
            passenger_pos = passenger.get("position")
            if passenger_pos:
                dist = self._manhattan_distance(position, passenger_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = passenger_pos
                    
        return nearest
    
    def _is_in_delivery_zone(self, position):
        """Check if the given position is in the delivery zone"""
        if not self.delivery_zone:
            return False
            
        delivery_pos = self.delivery_zone.get("position", (0, 0))
        delivery_width = self.delivery_zone.get("width", 40)
        delivery_height = self.delivery_zone.get("height", 40)
        
        x, y = position
        zone_x, zone_y = delivery_pos
        
        return (zone_x <= x < zone_x + delivery_width and 
                zone_y <= y < zone_y + delivery_height)
    
    def _find_path(self, start, end):
        """Use A* algorithm to find a path from start to end"""
        if not start or not end or not self.game_width or not self.game_height or not self.cell_size:
            return []
            
        # A* algorithm
        open_set = []
        closed_set = set()
        
        # Initialize with start node
        heapq.heappush(open_set, (0, start, []))  # (f_score, position, path)
        g_scores = {start: 0}
        
        while open_set:
            _, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            # If we've reached the end
            if current == end:
                return path + [end]
                
            closed_set.add(current)
            
            # Check all neighbors
            for dx, dy in BASE_DIRECTIONS:
                nx, ny = current[0] + dx * self.cell_size, current[1] + dy * self.cell_size
                neighbor = (nx, ny)
                
                # Skip if out of bounds
                if (nx < 0 or nx >= self.game_width or 
                    ny < 0 or ny >= self.game_height):
                    continue
                    
                # Skip if neighbor is in closed set
                if neighbor in closed_set:
                    continue
                    
                # Skip if neighbor has a train or wagon
                if self._is_occupied(neighbor):
                    continue
                    
                # Calculate scores
                g_score = g_scores[current] + self.cell_size
                h_score = self._manhattan_distance(neighbor, end)
                f_score = g_score + h_score
                
                # If this is a better path to the neighbor
                if neighbor not in g_scores or g_score < g_scores[neighbor]:
                    g_scores[neighbor] = g_score
                    new_path = path + [current]
                    heapq.heappush(open_set, (f_score, neighbor, new_path))
        
        # No path found
        return []
    
    def _is_occupied(self, position):
        """Check if a position is occupied by a train or wagon"""
        for train_name, train in self.all_trains.items():
            # Skip our own train
            if train_name == self.agent_name:
                continue
                
            # Check train position
            if train.get("position") == position:
                return True
                
            # Check wagon positions
            for wagon_pos in train.get("wagons", []):
                if wagon_pos == position:
                    return True
        
        return False
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_direction_to(self, from_pos, to_pos):
        """Get the direction needed to move from from_pos to to_pos"""
        if not from_pos or not to_pos or from_pos == to_pos:
            return None
            
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Align with grid
        if dx != 0:
            dx = dx // abs(dx) if dx != 0 else 0
        if dy != 0:
            dy = dy // abs(dy) if dy != 0 else 0
            
        # We can only move in one direction at a time
        if dx != 0 and dy != 0:
            # Choose the direction with the larger magnitude
            if abs(to_pos[0] - from_pos[0]) > abs(to_pos[1] - from_pos[1]):
                return (dx, 0)
            else:
                return (0, dy)
        
        return (dx, dy) if (dx, dy) in BASE_DIRECTIONS else None
    
    def _get_safe_random_direction(self, position):
        """Get a random direction that doesn't lead to immediate danger"""
        safe_directions = []
        
        for direction in BASE_DIRECTIONS:
            nx = position[0] + direction[0] * self.cell_size
            ny = position[1] + direction[1] * self.cell_size
            neighbor = (nx, ny)
            
            # Skip if out of bounds
            if (nx < 0 or nx >= self.game_width or 
                ny < 0 or ny >= self.game_height):
                continue
                
            # Skip if occupied
            if self._is_occupied(neighbor):
                continue
                
            safe_directions.append(direction)
        
        # If we have safe directions, choose one
        if safe_directions:
            return random.choice(safe_directions)
            
        # Otherwise, just choose any direction
        return random.choice(BASE_DIRECTIONS)