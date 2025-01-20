import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


from extract_sequence import locations
from extract_sequence import location_weights
from extract_sequence import devices
from extract_sequence import device_weights
from extract_sequence import networks
from extract_sequence import network_weights
from extract_sequence import possible_transactions

from agent import Agent  # Import the Agent class

class Activity(Agent):
    
    # Global dictionary for shared keys
    activity_DTYPE = {
        "agent_id": "int16",
        "timestamp": "datetime64[ns]",
        "behavior": "category",
        "initial_balance": "float32",
        "type": "category",
        "granted": "bool",
        "amount": "float32",
        "balance": "float32",
        #"merchant": "category",
        "location": "category",
        "device": "category",
        "network": "category",
        "compromised_device": "int8",
        "compromised_network": "int8",
        "fraud": "int8"
    }

    def __init__(self, agent_id, initial_balance, timestamp, behavior):
        # Initialize the parent class (Agent)
        super().__init__(agent_id, initial_balance)
        # activity-specific attributes
        self.timestamp = timestamp
        self.behavior = behavior
        self.type = type
        #self.balance = self.initial_balance
        self.granted = True 
        self.amount = 0
       #self.merchant =  extract_merchant(self.type) if self.amount !=0 else ''
        self.location = np.random.choice(locations, p=location_weights)
        self.device = np.random.choice(devices, p=device_weights)
        self.network = np.random.choice(networks, p=network_weights)
        self.compromised_device = self.is_compromised()
        self.compromised_network = self.is_compromised()
        self.fraud = 0 

    def is_compromised(self, probability=0.05):
        """
        Returns True if a device or network is compromised, based on the probability.
        Default probability of compromise is 5%.
        """
        return random.random() < probability

    def update_balance(self):
        """
        Update the agent's balance based on the transaction type and amount.
        """
        if self.amount>=0 or (self.amount<0 and self.balance>=abs(self.amount)):
            self.balance = self.balance + self.amount
        else:
            self.granted = False # Insufficient funds

    '''
    def extract_merchant(transaction):
    """
    Extract a random merchant based on the transaction type and probabilities.
    
    Args:
        transaction (str): The transaction type (e.g., 'purchase', 'deposit', 'withdrawal').
        
    Returns:
        str: The selected merchant.
    """
    if transaction not in TRANSACTION_TYPE_MERCHANTS:
        raise ValueError(f"Transaction type '{transaction}' is not recognized.")
    merchants, probabilities = zip(*TRANSACTION_TYPE_MERCHANTS[transaction]) # Unzip the dictionary for the given transaction to get an array of merchants and an array of relative probabilities
    merchant = np.random.choice(merchants, p=probabilities)
    return merchant
    '''