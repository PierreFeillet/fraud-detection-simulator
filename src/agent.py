import random
import pandas as pd
import numpy as np


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, agent_id, initial_balance, residence_country):
        self.agent_id = agent_id
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.residence_country = residence_country