import random
import pandas as pd
import numpy as np

from catalog import generate_transaction_amount, transaction_type, account_activity
from catalog import behavioral_catalog
from catalog import MERCHANT_TRANSACTION_TYPES
from catalog import merchants
from catalog import merchant_weights
from catalog import locations
from catalog import location_weights
from catalog import devices
from catalog import device_weights
from catalog import networks
from catalog import network_weights

# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, agent_id, balance):
        self.agent_id = agent_id
        self.balance = balance
        self.type = type=np.random.choice(["bank", "merchant", "client"], p=[0.2, 0.4, 0.4])

    # Define is_fraud() function
    def is_fraud(self, probability=0.05):
        """
        Returns True if a device or network is compromised, based on the probability.
        Default probability of compromise is 5%.
        """
        return random.random() < probability