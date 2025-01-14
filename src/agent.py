import random
import pandas as pd
import numpy as np


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, agent_id, balance):
        self.agent_id = agent_id
        self.balance = balance