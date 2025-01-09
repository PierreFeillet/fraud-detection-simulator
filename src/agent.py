import random
import pandas as pd
import numpy as np


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, agent_id, balance):
        self.agent_id = agent_id
        self.balance = balance
        self.type = type=np.random.choice(["bank", "merchant", "client"], p=[0.2, 0.4, 0.4])

    # Define is_compromised() function
    def is_compromised(self, probability=0.05):
        """
        Returns True if a device or network is compromised, based on the probability.
        Default probability of compromise is 5%.
        """
        return random.random() < probability