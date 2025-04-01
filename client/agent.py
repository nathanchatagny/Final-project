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

        # You can add any additional attributes here. For example:
        # self.some_attribute = None

        # You can ask the server to drop a wagon:
        # self.network.send_drop_wagon_request()

    def get_direction(self):
        """
        This method is regularly called by the client to get the next direction of the train.
        """
        return random.choice(BASE_DIRECTIONS) # Replace this with your own logic
