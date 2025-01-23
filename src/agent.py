import random
import pandas as pd
import numpy as np


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, real_id, virtual_id, is_fraudster, behavior, initial_balance, residence_country):
        self.real_id = real_id # Agent Real Identity 
        self.virtual_id = virtual_id # How the Agent look like to the system 
        self.is_fraudster = is_fraudster
        self.behavior = behavior
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.residence_country = residence_country