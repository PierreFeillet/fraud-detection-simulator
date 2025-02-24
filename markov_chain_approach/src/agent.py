import random
import pandas as pd
import numpy as np
import catalog


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, real_id, virtual_id, is_fraud, behavior, initial_time):
        self.real_id = real_id # Agent Real Identity 
        self.virtual_id = virtual_id # How the Agent look like to the system 
        self.is_fraud = is_fraud
        self.behavior = behavior
        self.initial_balance = round(random.uniform(1000, 100000), 2)
        self.balance = self.initial_balance
        self.initial_time = initial_time
        self.initial_country = np.random.choice(catalog.locations)
        self.agent_type = "traveler" if random.random() < 0.3 else "static" # chance of 30% to be a traveler
        self.visited_countries = {self.initial_country: 1.0}
